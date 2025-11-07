from sc2.ids.ability_id import AbilityId
from sc2.unit import Unit

from ares.consts import (
    UnitRole,
    UnitTreeQueryType,
    EngagementResult,
    VICTORY_CLOSE_OR_BETTER,
    LOSS_MARGINAL_OR_WORSE,
    WORKER_TYPES,
)

from ares import AresBot
from cython_extensions import cy_distance_to_squared, cy_unit_pending
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from sc2.units import Units

from bot.combat.base_combat import BaseCombat
from bot.combat.battle_cruiser_combat import BattleCruiserCombat
from bot.combat.ground_range_combat import GroundRangeCombat
from bot.combat.scv_proxy_builder import SCVProxyBuilder
from bot.consts import ProxySCVStatus, BIO_FORCES
from bot.openings.opening_base import OpeningBase
from bot.openings.bio import Bio

PATH_THRESHOLD: int = 100

STATIC_DEFENCE: set[UnitTypeId] = {
    UnitTypeId.BUNKER,
    UnitTypeId.PLANETARYFORTRESS,
    UnitTypeId.SPINECRAWLER,
    UnitTypeId.PHOTONCANNON,
}


class ProxyMarauder(OpeningBase):
    _proxy_complete: bool
    _proxy_location: Point2
    _bio: OpeningBase
    _battle_cruiser_combat: BaseCombat
    _ground_range_combat: BaseCombat
    _scv_proxy_builder: BaseCombat

    SQUAD_ENGAGE_THRESHOLD: set[EngagementResult] = VICTORY_CLOSE_OR_BETTER
    SQUAD_DISENGAGE_THRESHOLD: set[EngagementResult] = LOSS_MARGINAL_OR_WORSE

    def __init__(self):
        super().__init__()
        self._attack_started: bool = False
        # scv.tag: {"to_build": UnitTypeId, "pos": Point2, "ts": int, "status": str}
        self._proxy_tracker: dict = dict()
        self._squad_id_to_engage_tracker: dict = dict()

    @property
    def army_comp(self) -> dict:
        if self.ai.supply_workers < 26 and self.ai.time < 390.0:
            return {
                UnitTypeId.MARAUDER: {"proportion": 1.0, "priority": 0},
            }
        else:
            return {
                UnitTypeId.MARAUDER: {"proportion": 0.3, "priority": 0},
                UnitTypeId.MARINE: {"proportion": 0.5, "priority": 0},
                UnitTypeId.MEDIVAC: {"proportion": 0.2, "priority": 1},
            }

    @property
    def upgrade_list(self) -> list[UpgradeId]:
        if len(self.ai.townhalls) < 2:
            if not cy_unit_pending(self.ai, UnitTypeId.MARAUDER):
                return []
            return [UpgradeId.PUNISHERGRENADES]
        else:
            return [UpgradeId.PUNISHERGRENADES, UpgradeId.SHIELDWALL]

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)
        self._proxy_complete = False
        self._proxy_location = self._calculate_proxy_location()
        self._battle_cruiser_combat = BattleCruiserCombat(ai, ai.config, ai.mediator)
        self._ground_range_combat = GroundRangeCombat(ai, ai.config, ai.mediator)
        self._scv_proxy_builder = SCVProxyBuilder(ai, ai.config, ai.mediator)

        self._bio = Bio()
        await self._bio.on_start(ai)

    async def on_step(self) -> None:
        proxy_scvs_amount: int = 2 if not self._proxy_complete else 0
        if proxy_scvs := self._handle_proxy_scv_assignment(
            proxy_scvs_amount, self._proxy_location
        ):
            await self._handle_proxy_rax_construction(proxy_scvs)

        if (
            len(
                [
                    b
                    for b in self.ai.mediator.get_own_structures_dict[
                        UnitTypeId.BARRACKS
                    ]
                    if b.is_ready
                ]
            )
            >= 2
        ):
            self._proxy_complete = True

        if self.ai.build_order_runner.build_completed and (
            len(self.ai.mediator.get_own_structures_dict[UnitTypeId.BARRACKS]) >= 1
            or self.ai.time > 180.0
        ):
            self._macro()

        await self._bio.on_step()

    def on_unit_created(self, unit: Unit) -> None:
        if unit.type_id in BIO_FORCES:
            self.ai.mediator.assign_role(tag=unit.tag, role=UnitRole.ATTACKING)

    def _macro(self):
        num_rax: int = len(
            self.ai.mediator.get_own_structures_dict[UnitTypeId.BARRACKS]
        )
        self._generic_macro_plan(
            self.army_comp,
            self.ai.start_location,
            self.upgrade_list,
            add_hellions=False,
            add_upgrades=True,
            can_expand=self.ai.minerals > 500,
            freeflow_mode=True,
            upgrade_to_pfs=False,
            num_gas_buildings=0
            if num_rax < 1
            else (1 if len(self.ai.townhalls) < 2 else 2),
            num_one_base_workers=13
            if num_rax < 2
            else (16 if self.ai.supply_used < 22 else 19),
        )

    async def _handle_proxy_rax_construction(self, proxy_scvs: Units) -> None:
        for scv in proxy_scvs:
            tag: int = scv.tag
            if tag not in self._proxy_tracker:
                if placement := self.ai.mediator.request_building_placement(
                    base_location=self._proxy_location,
                    structure_type=UnitTypeId.BARRACKS,
                    closest_to=self.ai.enemy_start_locations[0],
                ):
                    self._proxy_tracker[tag] = {
                        "to_build": UnitTypeId.BARRACKS,
                        "pos": placement,
                        "ts": self.ai.time,
                        "status": ProxySCVStatus.Moving,
                    }
                continue

            current_status: ProxySCVStatus = self._proxy_tracker[tag]["status"]
            match current_status:
                case ProxySCVStatus.Moving:
                    target_pos: Point2 = self._proxy_tracker[tag]["pos"]
                    to_build: UnitTypeId = self._proxy_tracker[tag]["to_build"]
                    if (
                        cy_distance_to_squared(scv.position, target_pos) <= 25.0
                        and self.ai.tech_requirement_progress(to_build) >= 1.0
                        and self.ai.can_afford(to_build)
                    ):
                        stuctures: list[Unit] = [
                            s
                            for s in self.ai.structures
                            if cy_distance_to_squared(s.position, target_pos) < 9.0
                        ]
                        if len(stuctures) > 0:
                            scv.smart(stuctures[0])
                        else:
                            scv.build(to_build, target_pos)
                        self._proxy_tracker[tag]["status"] = ProxySCVStatus.Building
                    else:
                        scv.move(target_pos)
                case ProxySCVStatus.Building:
                    close_enemy_workers: Units = self.ai.mediator.get_units_in_range(
                        start_points=[scv.position],
                        distances=9.0,
                        query_tree=UnitTreeQueryType.EnemyGround,
                    )[0].filter(lambda u: u.type_id in WORKER_TYPES)
                    if not scv.is_constructing_scv:
                        self._proxy_tracker[tag]["status"] = ProxySCVStatus.Idle
                    elif close_enemy_workers:
                        scv(AbilityId.HALT)
                        self._proxy_tracker[tag]["status"] = ProxySCVStatus.Defending
                case ProxySCVStatus.Idle:
                    self.ai.mediator.assign_role(tag=tag, role=UnitRole.GATHERING)
                case ProxySCVStatus.Defending:
                    target_pos: Point2 = self._proxy_tracker[tag]["pos"]

                    close_enemy_workers: Units = self.ai.mediator.get_units_in_range(
                        start_points=[scv.position],
                        distances=11.0,
                        query_tree=UnitTreeQueryType.EnemyGround,
                    )[0].filter(lambda u: u.type_id in WORKER_TYPES)
                    if close_enemy_workers:
                        scv.attack(close_enemy_workers[0])
                    if (
                        not close_enemy_workers
                        or cy_distance_to_squared(scv.position, target_pos) > 100.0
                    ):
                        self._proxy_tracker[tag]["status"] = ProxySCVStatus.Moving
