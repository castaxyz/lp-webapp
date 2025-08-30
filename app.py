import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# --- solver ---
def solve_lp_2vars(c, constraints, sense="max"):
    c1, c2 = c
    points = []

    # intersecciones con ejes
    for (a1, a2, sign, b) in constraints:
        if a1 != 0: points.append((b/a1, 0))
        if a2 != 0: points.append((0, b/a2))

    # intersecciones entre restricciones
    for (a1, a2, s1, b1), (a3, a4, s2, b2) in combinations(constraints, 2):
        A = np.array([[a1, a2], [a3, a4]], dtype=float)
        B = np.array([b1, b2], dtype=float)
        if np.linalg.det(A) != 0:
            x, y = np.linalg.solve(A, B)
            points.append((x, y))

    # filtrar región factible
    feasible = []
    for (x, y) in points:
        if x >= -1e-6 and y >= -1e-6:
            factible = True
            for (a1, a2, sign, b) in constraints:
                val = a1*x + a2*y
                if sign == "<=" and val > b + 1e-6: factible = False
                if sign == ">=" and val < b - 1e-6: factible = False
                if sign == "=" and abs(val - b) > 1e-6: factible = False
            if factible:
                feasible.append((round(x,4), round(y,4)))

    feasible = list(set(feasible))
    results = [((x,y), c1*x+c2*y) for (x,y) in feasible]

    if not results:
        return None, None, []

    if sense == "max":
        opt_point, opt_value = max(results, key=lambda t: t[1])
    else:
        opt_point, opt_value = min(results, key=lambda t: t[1])

    return opt_point, opt_value, results

# --- interfaz web ---
st.title("Modelo de Programación Lineal (2 variables)")

# Función objetivo
st.subheader("Función objetivo")
c1 = st.number_input("Coeficiente de x1", value=250.0)
c2 = st.number_input("Coeficiente de x2", value=300.0)
sense = st.radio("Tipo de problema", ["max", "min"])

# Restricciones
st.subheader("Restricciones")
num_rest = st.number_input("Número de restricciones", min_value=1, max_value=10, value=2, step=1)

constraints = []
for i in range(num_rest):
    st.markdown(f"**Restricción {i+1}**")
    a1 = st.number_input(f"a1 (x1) en R{i+1}", value=20.0 if i==0 else 10.0, key=f"a1_{i}")
    a2 = st.number_input(f"a2 (x2) en R{i+1}", value=15.0, key=f"a2_{i}")
    sign = st.selectbox("Signo", ["<=", ">=", "="], index=0, key=f"sign_{i}")
    b = st.number_input(f"b en R{i+1}", value=600.0 if i==0 else 450.0, key=f"b_{i}")
    constraints.append((a1, a2, sign, b))

if st.button("Resolver"):
    opt_point, opt_value, results = solve_lp_2vars((c1,c2), constraints, sense)

    if not results:
        st.error("⚠️ No existe región factible.")
    else:
        st.success(f"Óptimo en {opt_point}, valor Z = {opt_value:.2f}")

        # Mostrar tabla de resultados
        st.write("### Vértices factibles y valores de Z")
        st.table([{"x1":x, "x2":y, "Z":z} for ((x,y),z) in results])

        # Graficar
        plt.figure(figsize=(6,6))
        x_vals = np.linspace(0, max(p[0] for (p,_) in results)*1.3+5, 200)

        # Restricciones
        for (a1,a2,sign,b) in constraints:
            if a2 != 0:
                y_vals = (b - a1*x_vals)/a2
                plt.plot(x_vals, y_vals, label=f"{a1}x1+{a2}x2{sign}{b}")
            else:
                x_const = b/a1
                plt.axvline(x=x_const, label=f"{a1}x1{sign}{b}")

        # Puntos factibles
        for (p,z) in results:
            plt.scatter(*p, color="blue")
            plt.text(p[0]+0.2, p[1]+0.2, f"Z={z:.0f}", fontsize=8)

        # Óptimo
        plt.scatter(*opt_point, color="red", s=80, label=f"Óptimo Z={opt_value:.0f}")
        plt.xlabel("x1"); plt.ylabel("x2"); plt.legend(); plt.grid(True)
        st.pyplot(plt)
