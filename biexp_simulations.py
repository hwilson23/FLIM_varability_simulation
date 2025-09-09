import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt


    return np, plt


@app.cell
def _():
    from multiexp_simulation import simulate_multi_exponential_decay
    return (simulate_multi_exponential_decay,)


@app.cell
def _(plt, simulate_multi_exponential_decay):
    num_photons = 50
    true_lifetimes = [1.1, 0]  # Two lifetime components (in ns)
    true_amplitudes = [1, 0]  # Relative amplitudes (70% and 30%)
    time_range = (0, 12.5)  # nanoseconds
    time_bins = 256
    # background_level = (num_photons*0.2)/256 #background counts per bin
    background_level = 0
    times_multi, counts_multi = simulate_multi_exponential_decay(
        num_photons=num_photons,
        lifetimes=true_lifetimes,
        fractions=true_amplitudes,
        time_range=time_range,
        time_bins=time_bins,
        background=background_level,
    )
    plt.plot(times_multi,counts_multi)
    return


@app.cell
def _(np):
    def simulate_multi_exponential_decay2(
        num_photons= 10_000, 
        lifetimes=(0.45,3.2), 
        fractions=(0.6,0.4), 
        time_range=(0,12.5), 
        time_bins=256, 
        background_lambda_poissonian=0
        ):
    
        min_time, max_time = time_range
        norm_amplitudes = np.array(fractions) / np.sum(fractions)
        component_photons = np.random.multinomial(num_photons, norm_amplitudes)
        all_photon_times = np.array([])
        for i, (lifetime, n_photons) in enumerate( zip(lifetimes,
                                                       component_photons)):
            if n_photons > 0:
                photon_times = np.random.exponential(scale=lifetime,
                                                     size=n_photons)
                all_photon_times = np.append(all_photon_times, photon_times)
        all_photon_times = all_photon_times[all_photon_times <= max_time]
        bin_edges = np.linspace(min_time, max_time, time_bins + 1)
        counts, _ = np.histogram(all_photon_times, bins=bin_edges)
        times = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        if background_lambda_poissonian > 0:
            bg_counts = np.random.poisson(background_lambda_poissonian, size=time_bins)
            counts = counts + bg_counts
        return times, counts

    return (simulate_multi_exponential_decay2,)


@app.cell
def _(mo):
    mo.md(r"""# Biexponential decay""")
    return


@app.cell
def _(mo):
    sl_num_photons= mo.ui.slider(0, 100000,10, label="$photons$",show_value=True,value=1000)
    sl_lifetime1 = mo.ui.slider(0, 3,0.1, label=r"$\tau_1$",show_value=True,value=0.45)
    sl_lifetime2 = mo.ui.slider(1, 8,0.2, label=r"$\tau_2$",show_value=True,value=3.2)
    sl_fractions = mo.ui.slider(0, 100, label=r"$\tau_1Fraction(%)$",show_value=True,value=50)
    sl_background = mo.ui.slider(0, 0.50,0.05, label=r"$bkg Poiss(\lambda)$",show_value=True,value=0.3)
    sl_num_photons,sl_lifetime1,sl_lifetime2,sl_fractions,sl_background
    return (
        sl_background,
        sl_fractions,
        sl_lifetime1,
        sl_lifetime2,
        sl_num_photons,
    )


@app.cell
def _(
    np,
    plt,
    simulate_multi_exponential_decay2,
    sl_background,
    sl_fractions,
    sl_lifetime1,
    sl_lifetime2,
    sl_num_photons,
):
    t,c = simulate_multi_exponential_decay2(num_photons=sl_num_photons.value,
                                            lifetimes=(sl_lifetime1.value,sl_lifetime2.value),
                                            background_lambda_poissonian=sl_background.value,
                                            fractions=( sl_fractions.value/100,1-( sl_fractions.value/100))
                                           )
    ax1 = plt.subplot(111)
    ax1.plot(t,c)
    ax2 = ax1.twinx()
    ax2.plot(t,np.log(c),'k',alpha=.5)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Flipper-TR

    - $\tau_1$ = $0.5 \dots 2 ns$
    - $\tau_2$ = $2.8 \dots 7 ns$ 
    """
    )
    return


@app.cell
def _(mo):

    sl_num_photons_fr= mo.ui.slider(0, 100000,10, label="$photons$",show_value=True,value=1000)
    sl_fractions_fr = mo.ui.slider(0, 100, label=r"$\tau_1Fraction(%)$",show_value=True,value=50)
    sl_background_fr = mo.ui.slider(0, 0.50,0.05, label=r"$bkg Poiss(\lambda)$",show_value=True,value=0.3)
    sl_num_photons_fr,sl_fractions_fr,sl_background_fr

    return sl_background_fr, sl_fractions_fr, sl_num_photons_fr


@app.cell
def _(
    plt,
    simulate_multi_exponential_decay2,
    sl_background_fr,
    sl_fractions_fr,
    sl_num_photons_fr,
):
    t1,c1 = simulate_multi_exponential_decay2(num_photons=sl_num_photons_fr.value,
                                            lifetimes=(0.5,2.8),
                                            background_lambda_poissonian=sl_background_fr.value,
                                            fractions=( sl_fractions_fr.value/100,1-( sl_fractions_fr.value/100))
                                           )
    t2,c2 = simulate_multi_exponential_decay2(num_photons=sl_num_photons_fr.value,
                                            lifetimes=(2.0,7),
                                            background_lambda_poissonian=sl_background_fr.value,
                                            fractions=( sl_fractions_fr.value/100,1-( sl_fractions_fr.value/100))
                                           )
    ax3 = plt.subplot(111)
    ax3.plot(t1,c1,'c',alpha=.9)
    ax4 = ax3.twinx()
    ax4.plot(t2,c2,'m',alpha=.8)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
