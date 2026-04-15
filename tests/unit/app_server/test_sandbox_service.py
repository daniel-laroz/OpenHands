"""Tests for SandboxService base class.

This module tests the SandboxService base class implementation, focusing on:
- pause_old_sandboxes method functionality
- Proper handling of pagination when searching sandboxes
- Correct filtering of running vs non-running sandboxes
- Proper sorting by creation time (oldest first)
- Error handling and edge cases
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from openhands.app_server.errors import MaxSandboxLimitReachedError
from openhands.app_server.sandbox.sandbox_models import (
    SandboxInfo,
    SandboxPage,
    SandboxStatus,
)
from openhands.app_server.sandbox.sandbox_service import SandboxService


class MockSandboxService(SandboxService):
    """Mock implementation of SandboxService for testing."""

    def __init__(self, max_num_sandboxes: int = 5):
        self.max_num_sandboxes = max_num_sandboxes
        self.search_sandboxes_mock = AsyncMock()
        self.get_sandbox_mock = AsyncMock()
        self.get_sandbox_by_session_api_key_mock = AsyncMock()
        self.start_sandbox_mock = AsyncMock()
        self.resume_sandbox_mock = AsyncMock()
        self.pause_sandbox_mock = AsyncMock()
        self.delete_sandbox_mock = AsyncMock()

    async def search_sandboxes(
        self, page_id: str | None = None, limit: int = 100
    ) -> SandboxPage:
        return await self.search_sandboxes_mock(page_id=page_id, limit=limit)

    async def get_sandbox(self, sandbox_id: str) -> SandboxInfo | None:
        return await self.get_sandbox_mock(sandbox_id)

    async def get_sandbox_by_session_api_key(
        self, session_api_key: str
    ) -> SandboxInfo | None:
        return await self.get_sandbox_by_session_api_key_mock(session_api_key)

    async def start_sandbox(
        self,
        sandbox_spec_id: str | None = None,
        sandbox_id: str | None = None,
        auto_pause_existing: bool = True,
    ) -> SandboxInfo:
        return await self.start_sandbox_mock(sandbox_spec_id, sandbox_id)

    async def resume_sandbox(
        self, sandbox_id: str, auto_pause_existing: bool = True
    ) -> bool:
        return await self.resume_sandbox_mock(sandbox_id)

    async def pause_sandbox(self, sandbox_id: str) -> bool:
        return await self.pause_sandbox_mock(sandbox_id)

    async def delete_sandbox(self, sandbox_id: str) -> bool:
        return await self.delete_sandbox_mock(sandbox_id)


def create_sandbox_info(
    sandbox_id: str,
    status: SandboxStatus,
    created_at: datetime,
    created_by_user_id: str | None = None,
    sandbox_spec_id: str = 'test-spec',
) -> SandboxInfo:
    """Helper function to create SandboxInfo objects for testing."""
    return SandboxInfo(
        id=sandbox_id,
        created_by_user_id=created_by_user_id,
        sandbox_spec_id=sandbox_spec_id,
        status=status,
        session_api_key='test-api-key' if status == SandboxStatus.RUNNING else None,
        created_at=created_at,
    )


@pytest.fixture
def mock_sandbox_service():
    """Fixture providing a mock sandbox service."""
    return MockSandboxService()


class TestEnforceSandboxLimit:
    """Test cases for the enforce_max_num_sandboxes_limit method."""

    @pytest.mark.asyncio
    async def test_cleanup_with_no_sandboxes(self, mock_sandbox_service):
        """Test enforcement when there are no sandboxes."""
        mock_sandbox_service.max_num_sandboxes = 5
        mock_sandbox_service.search_sandboxes_mock.return_value = SandboxPage(
            items=[], next_page_id=None
        )

        result = await mock_sandbox_service.enforce_max_num_sandboxes_limit(
            auto_pause_existing=True
        )

        assert result == []
        mock_sandbox_service.search_sandboxes_mock.assert_called_once_with(
            page_id=None, limit=100
        )
        mock_sandbox_service.pause_sandbox_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_within_limit(self, mock_sandbox_service):
        """Test enforcement when sandbox count is strictly within the limit."""
        mock_sandbox_service.max_num_sandboxes = 5
        now = datetime.now(timezone.utc)
        sandboxes = [
            create_sandbox_info('sb1', SandboxStatus.RUNNING, now - timedelta(hours=3)),
            create_sandbox_info('sb2', SandboxStatus.RUNNING, now - timedelta(hours=2)),
            create_sandbox_info('sb3', SandboxStatus.RUNNING, now - timedelta(hours=1)),
        ]

        mock_sandbox_service.search_sandboxes_mock.return_value = SandboxPage(
            items=sandboxes, next_page_id=None
        )

        result = await mock_sandbox_service.enforce_max_num_sandboxes_limit(
            auto_pause_existing=True
        )

        assert result == []
        mock_sandbox_service.pause_sandbox_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_exact_limit(self, mock_sandbox_service):
        """Test enforcement when sandbox count exactly equals the limit.
        Since we need room to start a NEW sandbox, hitting the limit means we must pause 1.
        """
        mock_sandbox_service.max_num_sandboxes = 3
        now = datetime.now(timezone.utc)
        sandboxes = [
            create_sandbox_info('sb1', SandboxStatus.RUNNING, now - timedelta(hours=3)),
            create_sandbox_info('sb2', SandboxStatus.RUNNING, now - timedelta(hours=2)),
            create_sandbox_info('sb3', SandboxStatus.RUNNING, now - timedelta(hours=1)),
        ]

        mock_sandbox_service.search_sandboxes_mock.return_value = SandboxPage(
            items=sandboxes, next_page_id=None
        )
        mock_sandbox_service.pause_sandbox_mock.return_value = True

        result = await mock_sandbox_service.enforce_max_num_sandboxes_limit(
            auto_pause_existing=True
        )

        # To free up capacity for the 3rd allowed sandbox, it pauses 1
        assert result == ['sb1']
        mock_sandbox_service.pause_sandbox_mock.assert_called_once_with('sb1')

    @pytest.mark.asyncio
    async def test_cleanup_exceeds_limit(self, mock_sandbox_service):
        """Test cleanup when sandbox count exceeds the limit."""
        mock_sandbox_service.max_num_sandboxes = 3
        now = datetime.now(timezone.utc)
        sandboxes = [
            create_sandbox_info(
                'sb1', SandboxStatus.RUNNING, now - timedelta(hours=5)
            ),  # oldest
            create_sandbox_info('sb2', SandboxStatus.RUNNING, now - timedelta(hours=4)),
            create_sandbox_info('sb3', SandboxStatus.RUNNING, now - timedelta(hours=3)),
            create_sandbox_info('sb4', SandboxStatus.RUNNING, now - timedelta(hours=2)),
            create_sandbox_info(
                'sb5', SandboxStatus.RUNNING, now - timedelta(hours=1)
            ),  # newest
        ]

        mock_sandbox_service.search_sandboxes_mock.return_value = SandboxPage(
            items=sandboxes, next_page_id=None
        )
        mock_sandbox_service.pause_sandbox_mock.return_value = True

        result = await mock_sandbox_service.enforce_max_num_sandboxes_limit(
            auto_pause_existing=True
        )

        # Capacity target: 3 - 1 = 2 kept running. (5 - 2 = 3 paused)
        assert result == ['sb1', 'sb2', 'sb3']
        assert mock_sandbox_service.pause_sandbox_mock.call_count == 3

    @pytest.mark.asyncio
    async def test_auto_pause_false_raises_error(self, mock_sandbox_service):
        """Test that if limit is reached and auto_pause_existing is False, it raises 429."""
        mock_sandbox_service.max_num_sandboxes = 2
        now = datetime.now(timezone.utc)
        sandboxes = [
            create_sandbox_info('sb1', SandboxStatus.RUNNING, now - timedelta(hours=2)),
            create_sandbox_info('sb2', SandboxStatus.RUNNING, now - timedelta(hours=1)),
        ]

        mock_sandbox_service.search_sandboxes_mock.return_value = SandboxPage(
            items=sandboxes, next_page_id=None
        )

        with pytest.raises(
            MaxSandboxLimitReachedError,
            match='You have reached the maximum number of running sandboxes',
        ):
            await mock_sandbox_service.enforce_max_num_sandboxes_limit(
                auto_pause_existing=False
            )

    @pytest.mark.asyncio
    async def test_cleanup_filters_non_running_sandboxes(self, mock_sandbox_service):
        """Test that cleanup only considers running sandboxes."""
        mock_sandbox_service.max_num_sandboxes = 2
        now = datetime.now(timezone.utc)
        sandboxes = [
            create_sandbox_info('sb1', SandboxStatus.RUNNING, now - timedelta(hours=5)),
            create_sandbox_info(
                'sb2', SandboxStatus.PAUSED, now - timedelta(hours=4)
            ),  # ignored
            create_sandbox_info('sb3', SandboxStatus.RUNNING, now - timedelta(hours=3)),
            create_sandbox_info(
                'sb4', SandboxStatus.ERROR, now - timedelta(hours=2)
            ),  # ignored
            create_sandbox_info('sb5', SandboxStatus.RUNNING, now - timedelta(hours=1)),
        ]

        mock_sandbox_service.search_sandboxes_mock.return_value = SandboxPage(
            items=sandboxes, next_page_id=None
        )
        mock_sandbox_service.pause_sandbox_mock.return_value = True

        result = await mock_sandbox_service.enforce_max_num_sandboxes_limit(
            auto_pause_existing=True
        )

        # 3 running, limit 2 -> target running 1. Pause 2 oldest running.
        assert len(result) == 2
        assert result == ['sb1', 'sb3']

    @pytest.mark.asyncio
    async def test_cleanup_with_pagination(self, mock_sandbox_service):
        """Test cleanup handles pagination correctly."""
        mock_sandbox_service.max_num_sandboxes = 2
        now = datetime.now(timezone.utc)

        page1_sandboxes = [
            create_sandbox_info('sb1', SandboxStatus.RUNNING, now - timedelta(hours=3)),
            create_sandbox_info('sb2', SandboxStatus.RUNNING, now - timedelta(hours=2)),
        ]
        page2_sandboxes = [
            create_sandbox_info('sb3', SandboxStatus.RUNNING, now - timedelta(hours=1)),
        ]

        def search_side_effect(page_id=None, limit=100):
            if page_id is None:
                return SandboxPage(items=page1_sandboxes, next_page_id='page2')
            elif page_id == 'page2':
                return SandboxPage(items=page2_sandboxes, next_page_id=None)

        mock_sandbox_service.search_sandboxes_mock.side_effect = search_side_effect
        mock_sandbox_service.pause_sandbox_mock.return_value = True

        result = await mock_sandbox_service.enforce_max_num_sandboxes_limit(
            auto_pause_existing=True
        )

        # 3 running, limit 2 -> pause 2 oldest.
        assert len(result) == 2
        assert result == ['sb1', 'sb2']
        assert mock_sandbox_service.search_sandboxes_mock.call_count == 2

    @pytest.mark.asyncio
    async def test_cleanup_handles_pause_failures_raises_error(
        self, mock_sandbox_service
    ):
        """Test cleanup raises 429 when enough pause operations fail to free capacity."""
        mock_sandbox_service.max_num_sandboxes = 2
        now = datetime.now(timezone.utc)
        sandboxes = [
            create_sandbox_info('sb1', SandboxStatus.RUNNING, now - timedelta(hours=4)),
            create_sandbox_info('sb2', SandboxStatus.RUNNING, now - timedelta(hours=3)),
            create_sandbox_info('sb3', SandboxStatus.RUNNING, now - timedelta(hours=2)),
            create_sandbox_info('sb4', SandboxStatus.RUNNING, now - timedelta(hours=1)),
        ]

        mock_sandbox_service.search_sandboxes_mock.return_value = SandboxPage(
            items=sandboxes, next_page_id=None
        )

        def pause_side_effect(sandbox_id):
            if sandbox_id in ('sb1', 'sb2'):
                return False  # Simulate failure for TWO sandboxes
            return True

        mock_sandbox_service.pause_sandbox_mock.side_effect = pause_side_effect

        # Need to pause 4 - (2 - 1) = 3 sandboxes. Since sb1 fails, we only pause 2.
        with pytest.raises(
            MaxSandboxLimitReachedError,
            match='Could not automatically pause enough older sandboxes to free up capacity',
        ):
            await mock_sandbox_service.enforce_max_num_sandboxes_limit(
                auto_pause_existing=True
            )

    @pytest.mark.asyncio
    async def test_legacy_pause_old_sandboxes(self, mock_sandbox_service):
        """Test the legacy bridge method ignores the param and proxies to enforce_max."""
        mock_sandbox_service.max_num_sandboxes = 2
        now = datetime.now(timezone.utc)
        sandboxes = [
            create_sandbox_info('sb1', SandboxStatus.RUNNING, now - timedelta(hours=2)),
            create_sandbox_info('sb2', SandboxStatus.RUNNING, now - timedelta(hours=1)),
        ]
        mock_sandbox_service.search_sandboxes_mock.return_value = SandboxPage(
            items=sandboxes, next_page_id=None
        )
        mock_sandbox_service.pause_sandbox_mock.return_value = True

        # Pass 999 to prove it ignores the argument and uses max_num_sandboxes=2
        result = await mock_sandbox_service.pause_old_sandboxes(max_num_sandboxes=999)
        assert len(result) == 1
        assert result == ['sb1']


class TestValidateSandboxLimit:
    """Test cases for the validate_sandbox_limit method (Fast Admission Control)."""

    @pytest.mark.asyncio
    async def test_validate_sandbox_limit_auto_pause_true(self, mock_sandbox_service):
        """Should return immediately without DB calls if auto_pause_existing is True."""
        await mock_sandbox_service.validate_sandbox_limit(auto_pause_existing=True)
        mock_sandbox_service.search_sandboxes_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_validate_sandbox_limit_exceeds_raises(self, mock_sandbox_service):
        """Should raise 429 if auto_pause is False and we hit the limit."""
        mock_sandbox_service.max_num_sandboxes = 1
        now = datetime.now(timezone.utc)
        mock_sandbox_service.search_sandboxes_mock.return_value = SandboxPage(
            items=[create_sandbox_info('sb1', SandboxStatus.RUNNING, now)],
            next_page_id=None,
        )

        with pytest.raises(
            MaxSandboxLimitReachedError,
            match='You have reached the maximum number of running sandboxes',
        ):
            await mock_sandbox_service.validate_sandbox_limit(auto_pause_existing=False)

    @pytest.mark.asyncio
    async def test_validate_sandbox_limit_resume_paused_checks_limit(
        self, mock_sandbox_service
    ):
        """Should raise 429 when trying to resume a paused sandbox while max running are hit."""
        mock_sandbox_service.max_num_sandboxes = 1
        now = datetime.now(timezone.utc)

        # We want to resume 'sb2', which is currently paused.
        mock_sandbox_service.get_sandbox_mock.return_value = create_sandbox_info(
            'sb2', SandboxStatus.PAUSED, now
        )

        # But 'sb1' is already taking up the 1 slot limit.
        mock_sandbox_service.search_sandboxes_mock.return_value = SandboxPage(
            items=[create_sandbox_info('sb1', SandboxStatus.RUNNING, now)],
            next_page_id=None,
        )

        with pytest.raises(MaxSandboxLimitReachedError):
            await mock_sandbox_service.validate_sandbox_limit(
                sandbox_id='sb2', auto_pause_existing=False
            )

    @pytest.mark.asyncio
    async def test_validate_sandbox_limit_resume_already_running_skips_check(
        self, mock_sandbox_service
    ):
        """Should skip the global quota check if the requested sandbox is ALREADY running."""
        mock_sandbox_service.max_num_sandboxes = 1
        now = datetime.now(timezone.utc)

        # We are pinging an already running sandbox.
        mock_sandbox_service.get_sandbox_mock.return_value = create_sandbox_info(
            'sb1', SandboxStatus.RUNNING, now
        )

        await mock_sandbox_service.validate_sandbox_limit(
            sandbox_id='sb1', auto_pause_existing=False
        )
        mock_sandbox_service.search_sandboxes_mock.assert_not_called()
