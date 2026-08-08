"""EventRouter resync escape hatch: a stalled router asks for a re-election.

Covers exo-rkllama#37: pods restarted at different times leave routers synced
to sessions whose master no longer exists. Such a router either nacks the same
missing index forever, or watches another session's events stream past its
session filter. Both stalls must trigger a local ConnectionMessage so the node
re-runs its master election and converges onto the live session.

Also covers exo-rkllama#38, the uplink mirror of those downlink stalls: a
node whose local events are never indexed by the master (it keeps receiving
global events, but the master discards or never hears its uplink) retransmits
the same events forever and silently never appears in cluster state. After
enough retransmissions of one event the router must fire the same re-election
nudge.
"""

import pytest
from anyio import create_task_group, fail_after, sleep

from exo.routing.connection_message import ConnectionMessage
from exo.routing.event_router import EventRouter
from exo.shared.types.commands import ForwarderCommand, RequestEventLog
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.events import (
    GlobalForwarderEvent,
    LocalForwarderEvent,
    TestEvent,
)
from exo.utils.channels import channel

OUR_MASTER = NodeId("ourmaster")
FOREIGN_MASTER = NodeId("foreignmaster")
OUR_SESSION = SessionId(master_node_id=OUR_MASTER, election_clock=0)
FOREIGN_SESSION = SessionId(master_node_id=FOREIGN_MASTER, election_clock=1)


class Harness:
    def __init__(
        self,
        *,
        nack_threshold: int = 8,
        foreign_threshold: int = 100,
        unacked_threshold: int = 12,
    ) -> None:
        self.command_sender, self.command_receiver = channel[ForwarderCommand]()
        self.global_sender, self.global_receiver = channel[GlobalForwarderEvent]()
        self.local_sender, self.local_receiver = channel[LocalForwarderEvent]()
        self.resync_sender, self.resync_receiver = channel[ConnectionMessage]()
        self.router = EventRouter(
            OUR_SESSION,
            command_sender=self.command_sender,
            external_inbound=self.global_receiver,
            external_outbound=self.local_sender,
            resync_sender=self.resync_sender,
            resync_nack_threshold=nack_threshold,
            resync_foreign_threshold=foreign_threshold,
            resync_unacked_threshold=unacked_threshold,
        )

    def speed_up_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(self.router, "_retry_interval_seconds", 0.005)
        monkeypatch.setattr(self.router, "_retry_age_seconds", 0.0)


def event_for(session: SessionId, origin: NodeId, idx: int) -> GlobalForwarderEvent:
    return GlobalForwarderEvent(
        origin_idx=idx, origin=origin, session=session, event=TestEvent()
    )


@pytest.mark.anyio
async def test_foreign_session_stream_triggers_reelection() -> None:
    h = Harness(foreign_threshold=10)
    async with create_task_group() as tg:
        tg.start_soon(h.router.run)
        for idx in range(10):
            await h.global_sender.send(event_for(FOREIGN_SESSION, FOREIGN_MASTER, idx))
        with fail_after(2):
            nudge = (await h.resync_receiver.receive_at_least(1))[0]
        assert nudge == ConnectionMessage(connected=True)
        h.router.shutdown()


@pytest.mark.anyio
async def test_unanswered_nacks_trigger_reelection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = Harness(nack_threshold=3)
    monkeypatch.setattr(h.router, "_nack_base_seconds", 0.001)
    monkeypatch.setattr(h.router, "_nack_cap_seconds", 0.002)
    requests: list[ForwarderCommand] = []

    async def drain_commands() -> None:
        with h.command_receiver as commands:
            async for command in commands:
                requests.append(command)

    async def feed_gapped_events() -> None:
        # A gap the master never fills: each arrival arms at most one nack,
        # so attempts accumulate across arrivals without ever resetting.
        for idx in range(200):
            await h.global_sender.send(event_for(OUR_SESSION, OUR_MASTER, 50 + idx))
            await sleep(0.005)

    async with create_task_group() as tg:
        tg.start_soon(h.router.run)
        tg.start_soon(drain_commands)
        tg.start_soon(feed_gapped_events)
        with fail_after(5):
            nudge = (await h.resync_receiver.receive_at_least(1))[0]
        assert nudge == ConnectionMessage(connected=True)
        assert any(isinstance(r.command, RequestEventLog) for r in requests)
        tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_unacked_local_events_trigger_reelection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = Harness(unacked_threshold=3)
    h.speed_up_retries(monkeypatch)
    sender = h.router.sender()
    async with create_task_group() as tg:
        tg.start_soon(h.router.run)
        # The master never echoes this event back as indexed, so retransmits
        # accumulate until the uplink detector fires.
        await sender.send(TestEvent())
        with fail_after(5):
            nudge = (await h.resync_receiver.receive_at_least(1))[0]
        assert nudge == ConnectionMessage(connected=True)
        h.router.shutdown()


@pytest.mark.anyio
async def test_indexed_local_event_resets_unacked_detector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = Harness(unacked_threshold=3)
    h.speed_up_retries(monkeypatch)
    sender = h.router.sender()
    async with create_task_group() as tg:
        tg.start_soon(h.router.run)
        await sender.send(TestEvent())
        with fail_after(2):
            sent = (await h.local_receiver.receive_at_least(1))[0]
        # The master indexes the event; the ack must clear the retry counter
        # so no re-election fires afterwards.
        await h.global_sender.send(
            GlobalForwarderEvent(
                origin_idx=0, origin=OUR_MASTER, session=OUR_SESSION, event=sent.event
            )
        )
        with fail_after(2):
            while h.router.out_for_delivery:
                await sleep(0.01)
        assert h.router._unacked_retries == {}  # pyright: ignore[reportPrivateUsage]
        await sleep(0.1)
        assert h.resync_receiver.collect() == []
        h.router.shutdown()


@pytest.mark.anyio
async def test_in_session_progress_resets_both_detectors() -> None:
    h = Harness(foreign_threshold=6)
    internal = h.router.receiver()
    async with create_task_group() as tg:
        tg.start_soon(h.router.run)
        for idx in range(5):
            await h.global_sender.send(event_for(FOREIGN_SESSION, FOREIGN_MASTER, idx))
        for idx in range(3):
            await h.global_sender.send(event_for(OUR_SESSION, OUR_MASTER, idx))
        with fail_after(2):
            delivered = await internal.receive_at_least(3)
        assert [d.idx for d in delivered] == [0, 1, 2]
        for idx in range(5, 10):
            await h.global_sender.send(event_for(FOREIGN_SESSION, FOREIGN_MASTER, idx))
        await sleep(0.05)
        assert h.resync_receiver.collect() == []
        h.router.shutdown()
