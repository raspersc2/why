import importlib
from typing import Any, Optional

from ares import AresBot
from ares.behaviors.macro.mining import Mining
from ares.consts import UnitRole
from cython_extensions import cy_distance_to_squared
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.unit import Unit
from sc2.units import Units


def _to_snake(name: str) -> str:
    # Convert e.g. "OneBaseTempest" -> "one_base_tempest"
    out = []
    for i, c in enumerate(name):
        if i > 0:
            prev = name[i - 1]
            nxt = name[i + 1] if i + 1 < len(name) else ""
            if c.isupper() and (
                (not prev.isupper())  # lower->Upper
                or (prev.isupper() and nxt and not nxt.isupper())  # UPPER->UpperLower
            ):
                out.append("_")
        out.append(c.lower())
    return "".join(out)


class MyBot(AresBot):
    def __init__(self, game_step_override: Optional[int] = None):
        """Initiate custom bot

        Parameters
        ----------
        game_step_override :
            If provided, set the game_step to this value regardless of how it was
            specified elsewhere
        """
        super().__init__(game_step_override)
        self.opening_handler: Optional[Any] = None
        self.opening_chat_tag: bool = False
        self._switched_to_prevent_tie: bool = False

    def load_opening(self, opening_name: str) -> None:
        """Load opening from bot.openings.<snake_case> with class <PascalCase>"""
        module_path = f"bot.openings.{_to_snake(opening_name)}"
        module = importlib.import_module(module_path)
        opening_cls = getattr(module, opening_name, None)
        if opening_cls is None:
            raise ImportError(
                f"Opening class '{opening_name}' not found in '{module_path}'"
            )
        self.opening_handler = opening_cls()

    async def on_start(self) -> None:
        await super(MyBot, self).on_start()
        # Ares has initialized BuildOrderRunner at this point
        try:
            self.load_opening(self.build_order_runner.chosen_opening)
            if hasattr(self.opening_handler, "on_start"):
                await self.opening_handler.on_start(self)
        except Exception as exc:
            print(f"Failed to load opening: {exc}")

    async def on_step(self, iteration: int) -> None:
        await super(MyBot, self).on_step(iteration)
        if self.supply_used < 1:
            await self.client.leave()
        self.register_behavior(Mining())

        self._mules()

        if self.opening_handler and hasattr(self.opening_handler, "on_step"):
            await self.opening_handler.on_step()

        if not self.opening_chat_tag and self.time > 5.0:
            await self.chat_send(
                f"Tag: {self.build_order_runner.chosen_opening}", team_only=True
            )
            self.opening_chat_tag = True

    async def on_unit_created(self, unit: Unit) -> None:
        await super(MyBot, self).on_unit_created(unit)
        if unit.type_id == UnitTypeId.REAPER:
            self.mediator.assign_role(tag=unit.tag, role=UnitRole.HARASSING_REAPER)

        if self.opening_handler and hasattr(self.opening_handler, "on_unit_created"):
            self.opening_handler.on_unit_created(unit)

    async def on_unit_took_damage(self, unit: Unit, amount_damage_taken: float) -> None:
        await super(MyBot, self).on_unit_took_damage(unit, amount_damage_taken)

        compare_health: float = max(50.0, unit.health_max * 0.09)
        if unit.health < compare_health:
            self.mediator.cancel_structure(structure=unit)

    def _mules(self):
        oc_id: UnitTypeId = UnitTypeId.ORBITALCOMMAND
        structures_dict: dict[
            UnitTypeId, list[Unit]
        ] = self.mediator.get_own_structures_dict
        for oc in [s for s in structures_dict[oc_id] if s.energy >= 50]:
            mfs: list[Unit] = [
                mf
                for mf in self.mineral_field
                if cy_distance_to_squared(mf.position, oc.position) < 100.0
            ]
            if mfs:
                mf: Unit = max(mfs, key=lambda x: x.mineral_contents)
                oc(AbilityId.CALLDOWNMULE_CALLDOWNMULE, mf)

    """
    Can use `python-sc2` hooks as usual, but make a call the inherited method in the superclass
    Examples:
    """

    #
    # async def on_end(self, game_result: Result) -> None:
    #     await super(MyBot, self).on_end(game_result)
    #
    #     # custom on_end logic here ...
    #
    # async def on_building_construction_complete(self, unit: Unit) -> None:
    #     await super(MyBot, self).on_building_construction_complete(unit)
    #
    #     # custom on_building_construction_complete logic here ...
    #
    # async def on_unit_created(self, unit: Unit) -> None:
    #     await super(MyBot, self).on_unit_created(unit)
    #
    #     # custom on_unit_created logic here ...
    #
    # async def on_unit_destroyed(self, unit_tag: int) -> None:
    #     await super(MyBot, self).on_unit_destroyed(unit_tag)
    #
    #     # custom on_unit_destroyed logic here ...
    #
    # async def on_unit_took_damage(self, unit: Unit, amount_damage_taken: float) -> None:
    #     await super(MyBot, self).on_unit_took_damage(unit, amount_damage_taken)
    #
    #     # custom on_unit_took_damage logic here ...
