from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
from ares import ManagerMediator
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import (
    AMove,
    DropCargo,
    KeepUnitSafe,
    PathUnitToTarget,
    PickUpCargo,
    ShootTargetInRange,
)
from ares.consts import UnitRole, UnitTreeQueryType
from sc2.ids.ability_id import AbilityId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.combat.base_combat import BaseCombat
from bot.consts import COMMON_UNIT_IGNORE_TYPES

if TYPE_CHECKING:
    from ares import AresBot


@dataclass
class GenericDrops(BaseCombat):
    """Execute behavior for medivac and dropping units.


    Parameters
    ----------
    ai : AresBot
        Bot object that will be running the game
    config : Dict[Any, Any]
        Dictionary with the data from the configuration file
    mediator : ManagerMediator
        Used for getting information from managers in Ares.
    """

    ai: "AresBot"
    config: dict
    mediator: ManagerMediator

    def execute(self, units: Units, **kwargs) -> None:
        """Execute the mine drop.

        Parameters
        ----------
        units : list[Unit]
            The units we want MedivacMineDrop to control.
        **kwargs :
            See below.

        Keyword Arguments
        -----------------
        medivac_tag_to_units_tracker : dict[list[int], dict]
            Tracker detailing medivac tag to unit tags.
            And target for the drop.

        """
        # no units assigned to mine drop currently.
        if not units:
            return

        air_grid: np.ndarray = self.mediator.get_air_grid
        ground_grid: np.ndarray = self.mediator.get_ground_grid
        medivac_tag_to_units_tracker: dict[int, dict] = kwargs[
            "medivac_tag_to_units_tracker"
        ]

        # we have the exact units, but we need to split them depending on precise job.
        unit_role_dict: dict[UnitRole, set[int]] = self.mediator.get_unit_role_dict

        for medivac_tag, tracker_info in medivac_tag_to_units_tracker.items():
            medivac: Optional[Unit] = self.ai.unit_tag_dict.get(medivac_tag, None)

            units_to_pickup: list[Unit] = [
                u
                for u in units
                if u.tag in tracker_info["tags"]
                and u.tag in unit_role_dict[UnitRole.DROP_UNITS_TO_LOAD]
            ]

            dropped_off_units: list[Unit] = [
                u
                for u in units
                if u.tag in unit_role_dict[UnitRole.DROP_UNITS_ATTACKING]
                and u.tag in tracker_info["tags"]
            ]

            if medivac and medivac_tag in unit_role_dict[UnitRole.DROP_SHIP]:
                self._handle_medivac_dropping_units(
                    medivac, units_to_pickup, air_grid, tracker_info["target"]
                )
            self._handle_units_to_pickup(units_to_pickup, medivac, ground_grid)
            self._handle_dropped_units(
                ground_grid,
                dropped_off_units,
                medivac,
                tracker_info["target"],
                tracker_info["healing"],
            )

    def _handle_medivac_dropping_units(
        self,
        medivac: Unit,
        units_to_pickup: list[Unit],
        air_grid: np.ndarray,
        target: Point2,
    ) -> None:
        """Control medivacs involvement.

        Parameters
        ----------
        medivac :
            The medivac to control.
        units_to_pickup :
            The mines this medivac should carry.
        air_grid :
            Pathing grid this medivac can path on.
        target :
            Where should this medivac drop mines?
        """

        # can speed boost, do that and ignore other actions till next step
        if (
            medivac.is_moving
            and AbilityId.EFFECT_MEDIVACIGNITEAFTERBURNERS in medivac.abilities
        ):
            medivac(AbilityId.EFFECT_MEDIVACIGNITEAFTERBURNERS)
            return

        # initiate a new mine drop maneuver
        medivac_drop: CombatManeuver = CombatManeuver()

        # first priority is picking up units
        medivac_drop.add(
            PickUpCargo(
                unit=medivac,
                grid=air_grid,
                pickup_targets=units_to_pickup,
                cargo_switch_to_role=UnitRole.DROP_UNITS_ATTACKING,
            )
        )

        # path to target
        medivac_drop.add(
            PathUnitToTarget(
                unit=medivac,
                grid=air_grid,
                target=target,
                success_at_distance=4.0,
            )
        )
        # drop off the units
        medivac_drop.add(DropCargo(unit=medivac, target=medivac.position))
        medivac_drop.add(KeepUnitSafe(unit=medivac, grid=air_grid))

        # register the behavior so it will be executed.
        self.ai.register_behavior(medivac_drop)

    def _handle_units_to_pickup(
        self, units: list[Unit], medivac: Optional[Unit], ground_grid: np.ndarray
    ) -> None:
        """Control units waiting rescue.

        Parameters
        ----------
        units :
            Units this method should control.
        medivac :
            Medivac that could possibly pick these units up.
        ground_grid :
            Pathing grid these mines can path on.
        """
        for unit in units:
            if medivac:
                unit.move(medivac.position)
            else:
                self.mediator.assign_role(
                    tag=unit.tag, role=UnitRole.DROP_UNITS_ATTACKING
                )

    def _handle_dropped_units(
        self,
        grid: np.ndarray,
        units: list[Unit],
        medivac: Unit,
        target: Point2,
        healing: bool,
    ) -> None:
        """Control mines that've recently been dropped off.

        Parameters
        ----------
        units :
            Units this method should control.
        """
        if len(units) == 0:
            return
        near_enemy: dict[int, Units] = self.mediator.get_units_in_range(
            start_points=units,
            distances=13,
            query_tree=UnitTreeQueryType.AllEnemy,
            return_as_dict=True,
        )
        for unit in units:
            close_enemy: list[Unit] = [
                u
                for u in near_enemy[unit.tag]
                if (not u.is_cloaked or u.is_cloaked and u.is_revealed)
                and (not u.is_burrowed or u.is_burrowed and u.is_visible)
                and not u.is_memory
                and not u.is_snapshot
                and u.type_id not in COMMON_UNIT_IGNORE_TYPES
            ]
            maneuver: CombatManeuver = CombatManeuver()
            if healing:
                maneuver.add(KeepUnitSafe(unit=unit, grid=grid))
            else:
                maneuver.add(ShootTargetInRange(unit, close_enemy))
                maneuver.add(KeepUnitSafe(unit=unit, grid=grid))
                maneuver.add(AMove(unit, target))
            self.ai.register_behavior(maneuver)
