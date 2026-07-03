from ares import AresBot
from bot.openings.opening_base import OpeningBase
from bot.openings.worker_rush import WorkerRush


class WorkerRushFast(OpeningBase):
    _worker_rush: OpeningBase

    def __init__(self):
        super().__init__()

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)
        self._worker_rush = WorkerRush()
        await self._worker_rush.on_start(ai)

    async def on_step(self) -> None:
        await self._worker_rush.on_step()



