from agentbench_frame.miracle.agent_bridge import MiracleAgent


class CandidateAgent(MiracleAgent):
    def choose_cards(self, camp):
        return {
            "artifacts": ["HolyLight"],
            "creatures": ["Archer", "Swordsman", "BlackBat"],
        }

    def act(self, obs):
        return {"operation_type": "endround", "operation_parameters": {}}
