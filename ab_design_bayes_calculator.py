import streamlit as st
import numpy as np
from scipy.stats import beta
import math
import matplotlib.pyplot as plt

# --- Базовые функции ---
def simulate_betas(a_C, b_C, a_T, b_T):
    control_simulation = np.random.beta(a_C, b_C, size=10000)
    treatment_simulation = np.random.beta(a_T, b_T, size=10000)
    return control_simulation, treatment_simulation

def compute_expected_loss(control_simulation, treatment_simulation):
    loss_control = [max(j - i, 0) for i, j in zip(control_simulation, treatment_simulation)]
    loss_treatment = [max(i - j, 0) for i, j in zip(control_simulation, treatment_simulation)]
    return round(np.mean(loss_control), 4), round(np.mean(loss_treatment), 4)

def compute_ctb(control_simulation, treatment_simulation):
    treatment_won = [i <= j for i, j in zip(control_simulation, treatment_simulation)]
    control_won = [j <= i for i, j in zip(control_simulation, treatment_simulation)]
    return round(np.mean(treatment_won), 4), round(np.mean(control_won), 4)

def do_bayes_calculation(base_conv, relative_lift, minimum_ctb, acceptable_relative_loss, count_daily, historical_conv_std):
    duration_in_days = 'not_defined'
    winner_group = 'not_defined'
    ctb_winner = 'not_defined'
    exp_loss_winner = 'not_defined'
    exp_loss_looser = 'not_defined'

    reached_target = False
    prior_a = round(base_conv * 10 + 1, 3)
    prior_b = round(10 + 1 - base_conv * 10, 3)

    for i in range(50, 30000, 50):
        imps_ctrl = imps_test = i
        imps_all = i * 2
        duration_in_days = math.ceil(imps_all / count_daily)
        convs_ctrl = i * base_conv
        convs_test = i * base_conv * (1 + relative_lift)

        # сглаживание +1
        a_C, b_C = prior_a + convs_ctrl + 1, prior_b + imps_ctrl - convs_ctrl + 1
        a_T, b_T = prior_a + convs_test + 1, prior_b + imps_test - convs_test + 1

        sim_control, sim_test = simulate_betas(a_C, b_C, a_T, b_T)
        ctb = compute_ctb(sim_control, sim_test)
        exp_loss = compute_expected_loss(sim_control, sim_test)
        ctb_A = ctb[1]
        ctb_B = ctb[0]
        exp_loss_B = exp_loss[1]
        exp_loss_A = exp_loss[0]

        if ctb_B > ctb_A:
            winner_group = 'Test'
            ctb_winner = ctb_B
            exp_loss_winner = exp_loss_B
            exp_loss_looser = exp_loss_A
        else:
            winner_group = 'Control'
            ctb_winner = ctb_A
            exp_loss_winner = exp_loss_A
            exp_loss_looser = exp_loss_B

        if ctb_winner >= minimum_ctb and exp_loss_winner <= acceptable_relative_loss * base_conv:
            return imps_ctrl, imps_all, duration_in_days

    return None, None, None

# --- Streamlit UI ---
st.title("📊 Байесовский калькулятор выборки для A/B-теста")

avg_day = st.number_input("Среднее количество элементов в день", value=1000)
base_metric = st.number_input("Базовая метрика (в долях, например 0.05)", value=0.05, format="%.4f")
effect_pct = st.number_input("Ожидаемый эффект в процентах (%)", value=10.0, step=0.1, format="%.1f")

# Автоматический пересчёт эффекта в п.п.
effect_pp = (effect_pct / 100) * base_metric
st.markdown(f"🔁 Это соответствует эффекту в **{effect_pp:.5f}** в абсолютных значениях.")

# Параметры по умолчанию
min_ctb = 0.95
acceptable_loss = 0.05
std_dev = 0.1

if st.button("🚀 Рассчитать"):
    relative_lift = effect_pct / 100
    n_group, n_total, duration = do_bayes_calculation(
        base_metric,
        relative_lift,
        min_ctb,
        acceptable_loss,
        avg_day,
        std_dev,
    )

    if n_group:
        st.success("✅ Расчёт завершён успешно:")
        st.write(f"- Количество элементов в группу: **{n_group}**")
        st.write(f"- Всего элементов: **{n_total}**")
        st.write(f"- Продолжительность эксперимента: **{duration} дней**")
    else:
        st.error("❌ Не удалось достичь заданных параметров CTB и допустимого риска.")
