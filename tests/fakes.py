"""Тестовые адаптеры, недоступные для выбора через публичный HTTP API."""

from etl_agent.application import Planner
from etl_agent.domain import NextAction
from etl_agent.exceptions import PlanningError


class FakePlanner(Planner):
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def propose(self, goal: str) -> NextAction:
        self.calls.append(goal)
        if "permission" in goal.lower():
            action = "Проверить права учётной записи загрузки на чтение источника."
        elif "формат" in goal.lower():
            action = "Сверить формат проблемного поля CSV со схемой загрузки."
        else:
            action = "Проверить наличие входного файла для указанного запуска ETL."
        return NextAction(action=action, rationale=f"Сначала нужно проверить сведения: {goal}")


class FailingPlanner(Planner):
    @staticmethod
    async def propose(goal: str) -> NextAction:
        raise PlanningError("private provider details must not escape")
