import asyncio
import logging
import time
from abc import ABC, abstractmethod

import httpx

from openhands.app_server.errors import MaxSandboxLimitReachedError, SandboxError
from openhands.app_server.sandbox.sandbox_models import (
    AGENT_SERVER,
    SandboxInfo,
    SandboxPage,
    SandboxStatus,
)
from openhands.app_server.services.injector import Injector
from openhands.app_server.utils.docker_utils import (
    replace_localhost_hostname_for_docker,
)
from openhands.sdk.utils.models import DiscriminatedUnionMixin
from openhands.sdk.utils.paging import page_iterator

_logger = logging.getLogger(__name__)

SESSION_API_KEY_VARIABLE = 'OH_SESSION_API_KEYS_0'
WEBHOOK_CALLBACK_VARIABLE = 'OH_WEBHOOKS_0_BASE_URL'
ALLOW_CORS_ORIGINS_VARIABLE = 'OH_ALLOW_CORS_ORIGINS_0'


class SandboxService(ABC):
    """Service for accessing sandboxes in which conversations may be run."""

    max_num_sandboxes: int

    @abstractmethod
    async def search_sandboxes(
        self,
        page_id: str | None = None,
        limit: int = 100,
    ) -> SandboxPage:
        """Search for sandboxes."""

    @abstractmethod
    async def get_sandbox(self, sandbox_id: str) -> SandboxInfo | None:
        """Get a single sandbox. Return None if the sandbox was not found."""

    @abstractmethod
    async def get_sandbox_by_session_api_key(
        self, session_api_key: str
    ) -> SandboxInfo | None:
        """Get a single sandbox by session API key. Return None if the sandbox was not found."""

    async def batch_get_sandboxes(
        self, sandbox_ids: list[str]
    ) -> list[SandboxInfo | None]:
        """Get a batch of sandboxes, returning None for any which were not found."""
        results = await asyncio.gather(
            *[self.get_sandbox(sandbox_id) for sandbox_id in sandbox_ids]
        )
        return results

    @abstractmethod
    async def start_sandbox(
        self,
        sandbox_spec_id: str | None = None,
        sandbox_id: str | None = None,
        auto_pause_existing: bool = True,
    ) -> SandboxInfo:
        """Begin the process of starting a sandbox.

        Return the info on the new sandbox. If no spec is selected, use the default.
        If sandbox_id is provided, it will be used as the sandbox identifier instead
        of generating a random one.
        """

    @abstractmethod
    async def resume_sandbox(
        self, sandbox_id: str, auto_pause_existing: bool = True
    ) -> bool:
        """Begin the process of resuming a sandbox.

        Return True if the sandbox exists and is being resumed or is already running.
        Return False if the sandbox did not exist.
        """

    async def _get_running_sandbox_ids_oldest_first(self) -> list[str]:
        running_sandboxes: list[SandboxInfo] = []
        async for sandbox in page_iterator(self.search_sandboxes, limit=100):
            if sandbox.status == SandboxStatus.RUNNING:
                running_sandboxes.append(sandbox)

        running_sandboxes.sort(key=lambda x: x.created_at)
        return [sandbox.id for sandbox in running_sandboxes]

    async def batch_pause_sandboxes(
        self, sandboxes_to_pause: list[str], num_to_pause: int
    ) -> list[str]:
        paused_sandbox_ids: list[str] = []
        for sandbox_id in sandboxes_to_pause:
            if len(paused_sandbox_ids) >= num_to_pause:
                break

            try:
                success = await self.pause_sandbox(sandbox_id)
                if success:
                    paused_sandbox_ids.append(sandbox_id)
            except Exception:
                # Continue trying to pause other sandboxes even if one fails
                pass

        return paused_sandbox_ids

    async def validate_sandbox_limit(
        self, sandbox_id: str | None = None, auto_pause_existing: bool = True
    ) -> None:
        if auto_pause_existing:
            return

        check_limit_for_resume = False
        if sandbox_id:
            sandbox_info = await self.get_sandbox(sandbox_id)
            check_limit_for_resume = (not sandbox_info) or (
                sandbox_info.status == SandboxStatus.PAUSED
            )

        if (not sandbox_id) or check_limit_for_resume:
            running_sandboxes = await self._get_running_sandbox_ids_oldest_first()
            num_running = len(running_sandboxes)

            if num_running >= self.max_num_sandboxes:
                raise MaxSandboxLimitReachedError(
                    detail=(
                        'You have reached the maximum number of running sandboxes '
                        f'(current={num_running}, limit={self.max_num_sandboxes}). '
                        'Stop or pause an existing sandbox and retry, or call again with '
                        'auto_pause_existing=true to allow automatically pausing older sandboxes/conversations.'
                    )
                )

    async def enforce_max_num_sandboxes_limit(
        self, auto_pause_existing: bool
    ) -> list[str]:
        """If auto_pause_existing, pause the oldest sandboxes if there are more than max_num_sandboxes_to_keep_running running.
        In a multi user environment, this will pause sandboxes only for the current user. else return 429."""

        running_sandboxes = await self._get_running_sandbox_ids_oldest_first()
        num_running = len(running_sandboxes)

        if num_running < self.max_num_sandboxes:
            return []

        if not auto_pause_existing:
            raise MaxSandboxLimitReachedError(
                detail=(
                    'You have reached the maximum number of running sandboxes '
                    f'(current={num_running}, limit={self.max_num_sandboxes}). '
                    'Stop or pause an existing sandbox and retry, or call again with '
                    'auto_pause_existing=true to allow automatically pausing older sandboxes/conversations.'
                )
            )

        num_to_pause = len(running_sandboxes) - (self.max_num_sandboxes - 1)
        # Stop the oldest sandboxes
        paused_sandbox_ids = await self.batch_pause_sandboxes(
            sandboxes_to_pause=running_sandboxes, num_to_pause=num_to_pause
        )

        if len(paused_sandbox_ids) < num_to_pause:
            raise MaxSandboxLimitReachedError(
                detail=(
                    'Could not automatically pause enough older sandboxes to free up capacity '
                    f'(Limit: {self.max_num_sandboxes}). '
                    'Please manually stop or pause an existing sandbox and try again.'
                )
            )

        return paused_sandbox_ids

    async def wait_for_sandbox_running(
        self,
        sandbox_id: str,
        timeout: int = 120,
        poll_interval: int = 2,
        httpx_client: httpx.AsyncClient | None = None,
    ) -> SandboxInfo:
        """Wait for a sandbox to reach RUNNING status with an alive agent server.

        This method polls the sandbox status until it reaches RUNNING state and
        optionally verifies the agent server is responding to health checks.

        Args:
            sandbox_id: The sandbox ID to wait for
            timeout: Maximum time to wait in seconds (default: 120)
            poll_interval: Time between status checks in seconds (default: 2)
            httpx_client: Optional httpx client for agent server health checks.
                If provided, will verify the agent server /alive endpoint responds
                before returning.

        Returns:
            SandboxInfo with RUNNING status and verified agent server

        Raises:
            SandboxError: If sandbox not found, enters ERROR state, or times out
        """
        start = time.time()
        while time.time() - start <= timeout:
            sandbox = await self.get_sandbox(sandbox_id)
            if sandbox is None:
                raise SandboxError(f'Sandbox not found: {sandbox_id}')

            if sandbox.status == SandboxStatus.ERROR:
                raise SandboxError(f'Sandbox entered error state: {sandbox_id}')

            if sandbox.status == SandboxStatus.RUNNING:
                # Optionally verify agent server is alive to avoid race conditions
                # where sandbox reports RUNNING but agent server isn't ready yet
                if httpx_client and sandbox.exposed_urls:
                    if await self._check_agent_server_alive(sandbox, httpx_client):
                        return sandbox
                    # Agent server not ready yet, continue polling
                else:
                    return sandbox

            await asyncio.sleep(poll_interval)

        raise SandboxError(f'Sandbox failed to start within {timeout}s: {sandbox_id}')

    async def _check_agent_server_alive(
        self, sandbox: SandboxInfo, httpx_client: httpx.AsyncClient
    ) -> bool:
        """Check if the agent server is responding to health checks.

        Args:
            sandbox: The sandbox info containing exposed URLs
            httpx_client: HTTP client to make the health check request

        Returns:
            True if agent server is alive, False otherwise
        """
        url = None
        try:
            agent_server_url = self._get_agent_server_url(sandbox)
            url = f'{agent_server_url.rstrip("/")}/alive'
            response = await httpx_client.get(url, timeout=5.0)
            return response.is_success
        except Exception as exc:
            _logger.debug(
                f'Agent server health check failed for sandbox {sandbox.id}'
                f'{f" at {url}" if url else ""}: {exc}'
            )
            return False

    def _get_agent_server_url(self, sandbox: SandboxInfo) -> str:
        """Get agent server URL from sandbox exposed URLs.

        Args:
            sandbox: The sandbox info containing exposed URLs

        Returns:
            The agent server URL

        Raises:
            SandboxError: If no agent server URL is found
        """
        if not sandbox.exposed_urls:
            raise SandboxError(f'No exposed URLs for sandbox: {sandbox.id}')

        for exposed_url in sandbox.exposed_urls:
            if exposed_url.name == AGENT_SERVER:
                return replace_localhost_hostname_for_docker(exposed_url.url)

        raise SandboxError(f'No agent server URL found for sandbox: {sandbox.id}')

    @abstractmethod
    async def pause_sandbox(self, sandbox_id: str) -> bool:
        """Begin the process of pausing a sandbox.

        Return True if the sandbox exists and is being paused or is already paused.
        Return False if the sandbox did not exist.
        """

    @abstractmethod
    async def delete_sandbox(self, sandbox_id: str) -> bool:
        """Begin the process of deleting a sandbox (which may involve stopping it).

        Return False if the sandbox did not exist.
        """

    async def pause_old_sandboxes(self, max_num_sandboxes: int) -> list[str]:
        """
        Implementation of the SandboxService abstract method.

        NOTE: This is a legacy bridge. 'enforce_max_num_sandboxes_limit' is the
        primary entry point for limit management in the Remote implementation
        as it handles the opt-out (429) logic.
        """
        # We ignore the passed max_num_sandboxes as enforce_max_num_sandboxes_limit uses self.max_num_sandboxes
        return await self.enforce_max_num_sandboxes_limit(auto_pause_existing=True)


class SandboxServiceInjector(DiscriminatedUnionMixin, Injector[SandboxService], ABC):
    pass
