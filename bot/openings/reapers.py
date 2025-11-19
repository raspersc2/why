from ares import AresBot
from ares.managers.squad_manager import UnitSquad
from cython_extensions import cy_center, cy_closest_to
from sc2.data import Race
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units
from src.ares.consts import (
    TOWNHALL_TYPES,
    WORKER_TYPES,
    EngagementResult,
    UnitRole,
    UnitTreeQueryType,
)

from bot.combat.base_combat import BaseCombat
from bot.combat.reaper_harass import ReaperHarass
from bot.consts import COMMON_UNIT_IGNORE_TYPES
from bot.openings.opening_base import OpeningBase


class Reapers(OpeningBase):
    _reaper_harass: BaseCombat
    reaper_harass_target: Point2

    def __init__(self):
        super().__init__()
        self.reaper_retreat_threshold: float = 0.6

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)
        self._reaper_harass = ReaperHarass(ai, ai.config, ai.mediator)
        self.reaper_harass_target = ai.enemy_start_locations[0]

    async def on_step(self) -> None:
        if (
            self.ai.build_order_runner.build_completed
            and self.ai.build_order_runner.chosen_opening == "Reapers"
        ):
            self._macro()
            target: Point2 = self.attack_target
            for hellion in self.ai.mediator.get_own_army_dict[UnitTypeId.HELLION]:
                hellion.attack(target)

        self._micro()

    @property
    def required_upgrades(self) -> list[UpgradeId]:
        return [
            UpgradeId.TERRANINFANTRYWEAPONSLEVEL1,
            UpgradeId.TERRANINFANTRYARMORSLEVEL1,
        ]

    def _macro(self):
        army_comp = {UnitTypeId.REAPER: {"proportion": 1.0, "priority": 0}}
        if self.ai.minerals > 500 and self.ai.vespene < 42:
            army_comp = {UnitTypeId.MARINE: {"proportion": 1.0, "priority": 0}}

        self._generic_macro_plan(
            army_comp,
            self.ai.start_location,
            self.required_upgrades,
            add_hellions=True,
            add_upgrades=len(self.ai.gas_buildings) >= 3,
        )

    def _micro(self):
        if reapers := self.ai.mediator.get_units_from_role(
            role=UnitRole.HARASSING_REAPER
        ):
            self._update_harass_target(reapers)
            self._execute_harass(reapers)

    def _execute_harass(self, reapers: Units) -> None:
        if reapers:
            squads: list[UnitSquad] = self.ai.mediator.get_squads(
                role=UnitRole.HARASSING_REAPER, squad_radius=7.5
            )
            for squad in squads:
                everything_near_reapers: Units = (
                    self.ai.mediator.get_units_in_range(
                        start_points=[squad.squad_position],
                        distances=12.0,
                        query_tree=UnitTreeQueryType.EnemyGround,
                        return_as_dict=False,
                    )[0]
                ).filter(
                    lambda u: u.type_id not in COMMON_UNIT_IGNORE_TYPES
                    or u.type_id == UnitTypeId.MULE
                )
                self._reaper_harass.execute(
                    squad.squad_units,
                    everything_near_reapers=everything_near_reapers,
                    harass_target=self.reaper_harass_target,
                    heal_threshold=self.reaper_retreat_threshold,
                )

    def _update_harass_target(self, reapers: Units) -> None:
        if not self.ai.mediator.get_enemy_roach_rushed and (
            main_threats := self.ai.mediator.get_main_ground_threats_near_townhall
        ):
            if self.ai.get_total_supply(main_threats) >= 2:
                self.reaper_harass_target = Point2(cy_center(main_threats))
                return

        if self.ai.enemy_race == Race.Terran:
            unfinished_bunkers = [
                s
                for s in self.ai.get_enemy_proxies(60.0, self.ai.start_location)
                if s.type_id == UnitTypeId.BUNKER and s.is_ready
            ]
            if len(unfinished_bunkers) > 0:
                self.reaper_harass_target = unfinished_bunkers[0].position
                return

        if self.ai.time < 150.0:
            self.reaper_harass_target = self.ai.enemy_start_locations[0]
            return

        if self.ai.time > 240.0 and not self.ai.enemy_structures:
            self.reaper_harass_target = self.attack_target
            return

        enemy_townhalls: list[Unit] = [
            th
            for th in self.ai.enemy_structures
            if th.type_id in TOWNHALL_TYPES and th.build_progress > 0.95
        ]
        best_engagement_result: EngagementResult = EngagementResult.LOSS_EMPHATIC
        potential_harass_target: Point2 = self.ai.enemy_start_locations[0]

        # this first if block allows the reaper to clear the map
        if self.ai.time > 300.0 and not enemy_townhalls and self.ai.enemy_structures:
            self.reaper_harass_target = cy_closest_to(
                self.ai.start_location, self.ai.enemy_structures
            ).position
            return

        positions_to_check: list[Point2] = [th.position for th in enemy_townhalls]
        positions_to_check.append(self.ai.enemy_start_locations[0])

        reaper_center: Point2 = Point2(cy_center(reapers))
        best_distance: float = float("inf")

        for position_to_check in positions_to_check:
            result: EngagementResult = self.ai.mediator.can_win_fight(
                own_units=reapers,
                enemy_units=self.ai.mediator.get_units_in_range(
                    start_points=[position_to_check],
                    distances=[15.0],
                    query_tree=UnitTreeQueryType.EnemyGround,
                )[0].filter(lambda u: u.type_id not in WORKER_TYPES),
            )

            # Only update if strictly better, or if equal then choose closer target
            if result.value > best_engagement_result.value:
                potential_harass_target = position_to_check
                best_engagement_result = result
                best_distance = reaper_center.distance_to(position_to_check)
            elif result.value == best_engagement_result.value:
                distance = reaper_center.distance_to(position_to_check)
                if distance < best_distance:
                    potential_harass_target = position_to_check
                    best_distance = distance

        self.reaper_harass_target = potential_harass_target
