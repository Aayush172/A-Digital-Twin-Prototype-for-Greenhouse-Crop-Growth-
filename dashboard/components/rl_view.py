"""Streamlit view for optional reinforcement-learning results."""

import json
from pathlib import Path

import streamlit as st


def render_rl_view(results_dir: Path, models_dir: Path, current_state) -> None:
    """Show saved PPO evaluation results and the current policy recommendation."""
    st.subheader("Reinforcement Learning Environmental Control")
    results_path = results_dir / "rl_training_results.json"
    model_path = models_dir / "ppo_greenhouse_agent.zip"

    if not results_path.exists() or not model_path.exists():
        st.info("Run the pipeline with --train-rl to generate PPO results and recommendations.")
        return

    results = json.loads(results_path.read_text(encoding="utf-8"))
    evaluation = results.get("evaluation", {})
    columns = st.columns(4)
    columns[0].metric("Mean reward", f"{evaluation.get('mean_reward', 0.0):.2f}")
    columns[1].metric("Mean growth (cm)", f"{evaluation.get('mean_growth', 0.0):.2f}")
    columns[2].metric("Mean episode length", f"{evaluation.get('mean_length', 0.0):.0f}")
    columns[3].metric("Evaluation episodes", str(len(results.get("training", {}).get("episode_rewards", []))))

    try:
        from optimization.agent import RLAgent
        from optimization.env import GreenhouseEnv

        environment = GreenhouseEnv(initial_state=current_state)
        agent = RLAgent.load(model_path, environment)
        observation, _ = environment.reset()
        action, _ = agent.model.predict(observation, deterministic=True)
        st.dataframe(
            {
                "Control": ["Heating", "Humidification", "CO2 injection", "Lighting"],
                "Recommended action": [float(value) for value in action],
            },
            hide_index=True,
        )
    except Exception as exc:
        st.warning(f"The saved policy could not be loaded: {exc}")