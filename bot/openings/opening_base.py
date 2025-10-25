from abc import ABCMeta, abstractmethod
from itertools import cycle

from ares import AresBot
from ares.behaviors.macro import (
    AutoSupply,
    BuildStructure,
    BuildWorkers,
    ExpansionController,
    GasBuildingController,
    MacroPlan,
    ProductionController,
    SpawnController,
    UpgradeCCs,
    UpgradeController,
)
from ares.cache import property_cache_once_per_frame
from ares.consts import UnitRole
from cython_extensions import cy_closest_to, cy_find_units_center_mass
from cython_extensions.geometry import cy_distance_to
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from sc2.units import Units

from bot.consts import ATTACK_TARGET_IGNORE


class OpeningBase(metaclass=ABCMeta):
    ai: AresBot

    def __init__(self):
        super().__init__()
        self.expansions_generator = None
        self.current_base_target: Point2 = Point2((0, 0))

    @abstractmethod
    async def on_start(self, ai: AresBot) -> None:
        self.ai = ai
        self.current_base_target = ai.enemy_start_locations[0]

    @abstractmethod
    async def on_step(self) -> None:
        pass

    def _generic_macro_plan(
        self,
        army_comp: dict,
        build_location: Point2,
        upgrades: list[UpgradeId],
        add_hellions: bool = True,
        add_upgrades: bool = True,
        freeflow_mode: bool = True,
        can_expand: bool = True,
        upgrade_to_pfs: bool = True,
        production_controller_enabled: bool = True,
        num_one_base_workers: int = 20,
    ) -> None:
        if (
            upgrade_to_pfs
            and not self.ai.structure_present_or_pending(UnitTypeId.ENGINEERINGBAY)
            and len(self.ai.townhalls) > 1
        ):
            self.ai.register_behavior(
                BuildStructure(build_location, UnitTypeId.ENGINEERINGBAY)
            )

        macro_plan: MacroPlan = MacroPlan()

        if production_controller_enabled:
            macro_plan.add(ProductionController(army_comp, build_location))

        macro_plan.add(AutoSupply(self.ai.start_location))
        macro_plan.add(GasBuildingController(100))
        if (
            upgrade_to_pfs
            and self.ai.mediator.get_own_structures_dict[UnitTypeId.ENGINEERINGBAY]
        ):
            macro_plan.add(UpgradeCCs(UnitTypeId.PLANETARYFORTRESS, prioritize=True))
        else:
            macro_plan.add(UpgradeCCs(UnitTypeId.ORBITALCOMMAND, prioritize=True))

        if add_upgrades:
            macro_plan.add(UpgradeController(upgrades, build_location))

        macro_plan.add(
            SpawnController(
                army_composition_dict=army_comp, freeflow_mode=freeflow_mode
            )
        )

        if add_hellions:
            macro_plan.add(
                SpawnController(
                    army_composition_dict={
                        UnitTypeId.HELLION: {"proportion": 1.0, "priority": 0}
                    },
                    freeflow_mode=freeflow_mode,
                )
            )

        num_workers: int = (
            num_one_base_workers
            if len(self.ai.townhalls) <= 1
            else (min(60, len(self.ai.townhalls) * 22))
        )

        macro_plan.add(BuildWorkers(num_workers))

        if can_expand:
            macro_plan.add(ExpansionController(100))
        self.ai.register_behavior(macro_plan)

    def _handle_proxy_probe_assignment(
        self, max_proxy_workers: int, proxy_location: Point2
    ) -> Units:
        proxy_workers: Units = self.ai.mediator.get_units_from_role(
            role=UnitRole.PROXY_WORKER
        )
        if len(proxy_workers) > max_proxy_workers:
            for worker in proxy_workers:
                self.ai.mediator.assign_role(tag=worker.tag, role=UnitRole.GATHERING)

        if len(proxy_workers) < max_proxy_workers:
            if worker := self.ai.mediator.select_worker(target_position=proxy_location):
                self.ai.mediator.assign_role(tag=worker.tag, role=UnitRole.PROXY_WORKER)

        return proxy_workers

    @property_cache_once_per_frame
    def attack_target(self) -> Point2:
        enemy_units: Units = self.ai.enemy_units.filter(
            lambda u: u.type_id not in ATTACK_TARGET_IGNORE
            and not u.is_flying
            and not u.is_cloaked
            and not u.is_hallucination
        )
        num_units: int = 0
        center_mass: Point2 = self.ai.start_location
        if enemy_units:
            center_mass, num_units = cy_find_units_center_mass(enemy_units, 12.5)
        enemy_structures: Units = self.ai.enemy_structures
        if num_units > 5:
            return Point2(center_mass)
        elif enemy_structures and self.ai.time > 120.0:
            return cy_closest_to(self.ai.start_location, enemy_structures).position
        elif (
            self.ai.time < 150.0
            or self.ai.state.visibility[self.ai.enemy_start_locations[0].rounded] == 0
        ):
            return self.ai.enemy_start_locations[0]
        else:
            # cycle through base locations
            if self.ai.is_visible(self.current_base_target):
                if not self.expansions_generator:
                    base_locations: list[Point2] = [
                        i for i in self.ai.expansion_locations_list
                    ]
                    self.expansions_generator = cycle(base_locations)

                self.current_base_target = next(self.expansions_generator)

            return self.current_base_target

    def _count_started_at_proxy(
        self, unit_id: UnitTypeId, target: Point2, radius: float = 18.0
    ) -> int:
        """Counts structures of type unit_id that are started (under construction or ready) near target."""
        structures = self.ai.mediator.get_own_structures_dict[unit_id]
        if len(structures) == 0:
            return 0
        return len(
            [s for s in structures if cy_distance_to(target, s.position) < radius]
        )

    def _next_build_target(
        self, plan: list[tuple[UnitTypeId, int]], target: Point2
    ) -> UnitTypeId | None:
        """Given a normalized plan, returns the next (unit_id, remaining_for_this_step).
        Ensures all previous steps are satisfied before moving on.
        """
        # cumulative approach: for each step, ensure that total built of that type at proxy
        # meets the sum of counts up to that step for that same type
        progress_per_type: dict[UnitTypeId, int] = {}
        for unit_id, count in plan:
            current_built = self._count_started_at_proxy(unit_id, target)
            already_required = progress_per_type.get(unit_id, 0)
            step_goal_total = already_required + count
            if current_built < step_goal_total:
                # still need to build for this step
                return unit_id
            progress_per_type[unit_id] = step_goal_total
        return None
