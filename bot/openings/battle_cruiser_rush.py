from ares import AresBot
from cython_extensions import cy_unit_pending
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from sc2.units import Units

from bot.combat.base_combat import BaseCombat
from bot.combat.battle_cruiser_combat import BattleCruiserCombat
from bot.combat.ground_range_combat import GroundRangeCombat
from bot.openings.opening_base import OpeningBase
from bot.openings.reapers import Reapers


class BattleCruiserRush(OpeningBase):
    _worker_rush_activated: bool
    _battle_cruiser_combat: BaseCombat
    _ground_range_combat: BaseCombat

    def __init__(self):
        super().__init__()
        self._attack_started: bool = False

    @property
    def army_comp(self) -> dict:
        return {
            UnitTypeId.BATTLECRUISER: {"proportion": 0.8, "priority": 0},
            UnitTypeId.MARINE: {"proportion": 0.2, "priority": 1},
        }

    @property
    def upgrade_list(self) -> list[UpgradeId]:
        # maybe later?
        # return [UpgradeId.BATTLECRUISERENABLESPECIALIZATIONS]
        return []

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)
        self._battle_cruiser_combat = BattleCruiserCombat(ai, ai.config, ai.mediator)
        self._ground_range_combat = GroundRangeCombat(ai, ai.config, ai.mediator)

    async def on_step(self) -> None:
        if self.ai.build_order_runner.build_completed:
            self._macro()

        attack_target: Point2 = self.attack_target
        bcs: Units = self.ai.mediator.get_own_army_dict[UnitTypeId.BATTLECRUISER]
        marines: Units = self.ai.mediator.get_own_army_dict[UnitTypeId.MARINE]
        if not self._attack_started and bcs:
            self._attack_started = True

        self._battle_cruiser_combat.execute(bcs, target=attack_target)

        marine_target = (
            self.ai.main_base_ramp.top_center
            if not self._attack_started
            else attack_target
        )
        self._ground_range_combat.execute(marines, target=marine_target)

    def _macro(self):
        pending_bcs: bool = cy_unit_pending(self.ai, UnitTypeId.BATTLECRUISER)
        self._generic_macro_plan(
            self.army_comp,
            self.ai.start_location,
            self.upgrade_list,
            add_hellions=False,
            add_upgrades=pending_bcs,
            can_expand=pending_bcs,
            freeflow_mode=True,
            upgrade_to_pfs=False,
        )
