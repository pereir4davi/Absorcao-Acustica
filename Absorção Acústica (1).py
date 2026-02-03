import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    # Unidades originais: Lambda em micrometros (um), Sigma em N.s/m4
    db_materiais = {
        "PSL32 (Sigma: 11496)":  {'phi': 0.96, 'alpha': 1.00, 'sigma': 11496, 'L': 277, 'Lp': 277},
        "PSE48 (Sigma: 11216)":  {'phi': 0.96, 'alpha': 1.00, 'sigma': 11216, 'L': 320, 'Lp': 320},
        "PSE64 (Sigma: 17083)":  {'phi': 0.89, 'alpha': 1.00, 'sigma': 17083, 'L': 197, 'Lp': 197},
        "PSE80 (Sigma: 23675)":  {'phi': 0.85, 'alpha': 1.00, 'sigma': 23675, 'L': 121, 'Lp': 121},
        "PSE96 (Sigma: 23385)":  {'phi': 0.92, 'alpha': 1.00, 'sigma': 23385, 'L': 78,  'Lp': 98},
        "PSR128 (Sigma: 34359)":  {'phi': 0.90, 'alpha': 1.00, 'sigma': 34359, 'L': 58,  'Lp': 73},
        "PSE48V1 (Sigma: 14869)":  {'phi': 0.96, 'alpha': 1.00, 'sigma': 14869, 'L': 234, 'Lp': 234},
        "PSE64V1 (Sigma: 18793)":  {'phi': 1.00, 'alpha': 1.00, 'sigma': 18793, 'L': 1652,'Lp': 1652},
        "PSE80V1 (Sigma: 24385)":  {'phi': 0.76, 'alpha': 1.00, 'sigma': 24385, 'L': 103, 'Lp': 103},
        "PSE96V1 (Sigma: 31333)":  {'phi': 0.94, 'alpha': 1.00, 'sigma': 31333, 'L': 85,  'Lp': 162},
        "PSR128V1 (Sigma: 27419)":  {'phi': 0.84, 'alpha': 1.00, 'sigma': 27419, 'L': 80,  'Lp': 124},
    }

    lista_nomes = list(db_materiais.keys())

    # interface Gráfica

    # Título do App
    mo.md("# 🎛️ Simulador Multicamada JCA").left()

    # Controles
    select_mat_A = mo.ui.dropdown(options=lista_nomes, value=lista_nomes[0], label="Material A (Fundo)")
    slider_d_A   = mo.ui.slider(start=0, stop=100, value=50, label="Espessura A (mm)")

    select_mat_B = mo.ui.dropdown(options=lista_nomes, value=lista_nomes[1], label="Material B (Frente)")
    slider_d_B   = mo.ui.slider(start=0, stop=100, value=25, label="Espessura B (mm)")


    # Organização na Sidebar (esquerda)
    sidebar_content = mo.vstack([
        mo.md("### ⚙️ Configurações"),
        select_mat_A, 
        slider_d_A,
        mo.md("---"),
        select_mat_B, 
        slider_d_B,
        mo.md("**Nota:** O material A fica encostado na parede rígida.")
    ])

    # Renderiza sidebar
    mo.sidebar(sidebar_content)
    return (
        db_materiais,
        mo,
        np,
        plt,
        select_mat_A,
        select_mat_B,
        slider_d_A,
        slider_d_B,
    )


@app.cell
def _(
    db_materiais,
    mo,
    np,
    plt,
    select_mat_A,
    select_mat_B,
    slider_d_A,
    slider_d_B,
):
    # Cálculos

    # Recupera valores escolhidos na interface gráfica
    nome_A = select_mat_A.value
    nome_B = select_mat_B.value
    espessura_A_user = slider_d_A.value / 1000.0  # Espessura Camada A [m]
    espessura_B_user = slider_d_B.value / 1000.0  # Espessura Camada B [m]

    # --- Constantes Físicas do Ar (Condições NTP) ---
    rho0  = 1.213       # Densidade do ar [kg/m³]
    c0    = 343.0       # Velocidade do som [m/s]
    P0    = 101325.0    # Pressão atmosférica [Pa]
    eta   = 1.84e-5     # Viscosidade dinâmica do ar [Pa.s]
    gamma = 1.4         # Razão de calores específicos
    Pr    = 0.71        # Número de Prandtl 
    Z0    = rho0 * c0   # Impedância acústica do ar [Pa.s/m]

    # Função Modelo JCA (Johnson-Champoux-Allard)
    def get_jca_params(dados_mat, w):
        # 1. Extração dos Parâmetros do Material
        phi   = dados_mat['phi']          # Porosidade (φ)
        alpha = dados_mat['alpha']        # Tortuosidade (α_inf)
        sigma = dados_mat['sigma']        # Resistividade ao Fluxo (σ) - [N.s/m⁴]
        L     = dados_mat['L'] * 1e-6     # Comp. Característico Viscoso (Λ) - [m]
        Lp    = dados_mat['Lp'] * 1e-6    # Comp. Característico Térmico (Λ') - [m]

        # Densidade Efetiva (ρ_eff)
        # Representa os efeitos inerciais e viscosos do fluido nos poros
        termo_viscoso = (sigma * phi) / (1j * w * rho0 * alpha)
        fator_correcao_visc = np.sqrt(1 + (4 * alpha**2 * eta * rho0 * w) / ((sigma * L * phi)**2))

        rho_eff = (alpha * rho0 / phi) * (1 + termo_viscoso * fator_correcao_visc)

        # Módulo de Compressibilidade Efetivo (K_eff)
        # Representa os efeitos térmicos (troca de calor fluido/estrutura)
        termo_termico = (8 * eta) / (1j * w * Pr * Lp**2 * rho0)
        fator_correcao_term = np.sqrt(1 + (1j * w * Pr * Lp**2 * rho0) / (16 * eta))

        k_eff = (gamma * P0 / phi) / (gamma - (gamma - 1) * (1 + termo_termico * fator_correcao_term)**(-1))

        # Propriedades de Propagação
        Zc = np.sqrt(rho_eff * k_eff)      # Impedância Característica do Material [Pa.s/m]
        kc = w * np.sqrt(rho_eff / k_eff)  # Número de Onda Complexo [rad/m]

        return Zc, kc

    # Função de calculo de Sistema (Matriz de Transferência)
    def calcular_sistema(mat1_nome, d1, mat2_nome, d2, frequencies):
        w = 2 * np.pi * frequencies  # Frequência angular (ω) [rad/s]

        # Passo 1: Camada de Fundo (Material 1)
        # Encostada na parede rígida
        if d1 > 0:
            Zc1, kc1 = get_jca_params(db_materiais[mat1_nome], w)
            # Impedância de superfície para material com fundo rígido: -j * Zc * cot(kd)
            Z_s1 = -1j * Zc1 * (1 / np.tan(kc1 * d1))
        else:
            Z_s1 = np.inf # Impedância infinita (Parede Rígida perfeita)

        # Passo 2: Camada da Frente (Material 2)
        # Encostada no ar, tendo a Camada 1 atrás
        if d2 > 0:
            Zc2, kc2 = get_jca_params(db_materiais[mat2_nome], w)
            tan_2 = np.tan(kc2 * d2)

            # Algoritmo de Transferência de Impedância
            if np.isinf(Z_s1).all(): 
                 # Se não tem camada 1, calcula como se a 2 estivesse na parede
                 Z_total = -1j * Zc2 * (1 / tan_2)
            else:
                 # Fórmula completa de transferência de impedância
                 num = Z_s1 + 1j * Zc2 * tan_2
                 den = Zc2 + 1j * Z_s1 * tan_2
                 Z_total = Zc2 * (num / den)
        else:
            # Se não tem camada 2, a superfície é a própria camada 1
            Z_total = Z_s1

        # Cálculo da Absorção
        # Caso especial: Sem material nenhum (reflexão total)
        if d1 == 0 and d2 == 0:
            return np.zeros_like(frequencies)

        # Coeficiente de Reflexão (R)
        R = (Z_total - Z0) / (Z_total + Z0)

        # Coeficiente de Absorção (Alpha)
        return 1 - np.abs(R)**2

    # Geração dos Dados (Plotagem e Tabela)
    freqs_plot = np.arange(50, 5001, 10)
    alpha_plot = calcular_sistema(nome_A, espessura_A_user, nome_B, espessura_B_user, freqs_plot)

    # Definição das frequências para a tabela (Bandas de Oitava/Terço)
    f_tab = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 
                      1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000])

    # Cálculos individuais para a tabela
    a_25 = calcular_sistema(nome_A, 0.025, nome_B, 0.000, f_tab)
    a_50 = calcular_sistema(nome_A, 0.050, nome_B, 0.000, f_tab)
    a_75 = calcular_sistema(nome_A, 0.075, nome_B, 0.000, f_tab)

    b_25 = calcular_sistema(nome_B, 0.025, nome_A, 0.000, f_tab)
    b_50 = calcular_sistema(nome_B, 0.050, nome_A, 0.000, f_tab)
    b_75 = calcular_sistema(nome_B, 0.075, nome_A, 0.000, f_tab)

    comb = calcular_sistema(nome_A, espessura_A_user, nome_B, espessura_B_user, f_tab)



    # COLOCANDO OS MATERIAIS NO GRAFICO
    # CÁLCULO DAS REFERÊNCIAS DE ESPESSURA TOTAL
    d_total = espessura_A_user + espessura_B_user # Espessura total
    # Se tivesse apenas material A
    alpha_ref_A = calcular_sistema(nome_A, d_total, nome_B, 0, freqs_plot)
    # Se tivesse apenas material B
    alpha_ref_B = calcular_sistema(nome_B, d_total, nome_A, 0, freqs_plot)



    # Tabela conforme o padrão Vibro Acustica (transposta)
    # Cabeçalho (Frequências)
    md_table = "| Frequência (Hz) | "
    for f in f_tab:
        md_table += f" **{int(f)}** |"
    md_table += "\n"

    md_table += "|---|" + "---|" * len(f_tab) + "\n"

    # Linhas
    linhas_dados = [
        (f"A (25mm)", a_25),
        (f"A (50mm)", a_50),
        (f"A (75mm)", a_75),
        (f"B (25mm)", b_25),
        (f"B (50mm)", b_50),
        (f"B (75mm)", b_75),
        ("**COMBINADO ATUAL**", comb)
    ]

    for titulo, dados in linhas_dados:
        md_table += f"| {titulo} |"
        for valor in dados:
            if titulo == "**COMBINADO ATUAL**":
                md_table += f" **{valor:.2f}** |"
            else:
                md_table += f" {valor:.2f} |"
        md_table += "\n"

    # Visualização Final
    plt.figure(figsize=(10, 5))
    plt.plot(freqs_plot, alpha_plot, label='Sistema Combinado', color='#2ecc71', linewidth=3)

    # CURVAS DE REFERENCIA
    # MATERIAL A SOZINHO
    plt.plot(freqs_plot, alpha_ref_A, 
             label=f"Ref: Só A ({int(d_total*1000)}mm)", 
             linestyle='--', color='#3498db', linewidth=1.5, alpha=0.7)

    # MATERIAL B SOZINHO
    plt.plot(freqs_plot, alpha_ref_B, 
             label=f"Ref: Só B ({int(d_total*1000)}mm)", 
             linestyle='--', color='#e74c3c', linewidth=1.5, alpha=0.7)
    plt.title(f"Absorção: {nome_A} ({int(espessura_A_user*1000)}mm) + {nome_B} ({int(espessura_B_user*1000)}mm)")
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("Absorção")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.legend()

    mo.vstack([
        plt.gca(),
        mo.md("### 📊 Tabela de Coeficientes"),
        mo.md(f"**Materiais:** A = *{nome_A}* | B = *{nome_B}*"),
        mo.md(md_table)
    ])
    return


if __name__ == "__main__":
    app.run()
