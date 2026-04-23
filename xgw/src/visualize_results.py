import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes
from sklearn.metrics import auc

from aliases import (
    AVG_PR_AUC,
    BRIER_SCORE,
    DENS_THRESH,
    INFLATION,
    LOG_LOSS,
    MATCH_THRESH,
    METHOD,
    METRIC,
    N_EDGES,
    PR_AUC,
    PREC,
    PROTEIN_U,
    PROTEIN_V,
    RECALL,
    SCENARIO,
    TOPO,
    VALUE,
    XVAL_ITER,
)
from utils import construct_composite_network


def summarize_composite_networks():
    """
    Prints a table of summarizing statistics of the two composite networks.
    """

    print("=====================================================")
    print("SUMMARIZING STATISTICS OF THE TWO COMPOSITE NETWORKS")
    print()

    df_orig_composite = construct_composite_network(dip=False)
    df_dip_composite = construct_composite_network(dip=True)

    def count_ppis_and_edges(df_composite: pl.DataFrame):
        num_ppis = df_composite.select(TOPO).filter(pl.col(TOPO) > 0).shape[0]
        num_edges = df_composite.select([PROTEIN_U, PROTEIN_V]).unique().shape[0]
        return num_ppis, num_edges

    orig_ppis, orig_edges = count_ppis_and_edges(df_orig_composite)
    dip_ppis, dip_edges = count_ppis_and_edges(df_dip_composite)

    df_table = pl.DataFrame(
        {
            "Composite Network": ["Original", "DIP"],
            "Num. of PPIs": [orig_ppis, dip_ppis],
            "Num. of edges": [orig_edges, dip_edges],
        }
    )

    print(df_table)
    print()


def remove_all_axes_labels(axes: list[Axes]):
    """
    Remove axes labels.
    """

    for ax in axes:
        ax.set_ylabel("")
        ax.set_xlabel("")


def plot_feat_importances():
    """
    Prints and plots feature importances.
    """

    print("=====================================================")
    print("FEATURE IMPORTANCES")
    print()

    df_orig_importances = pl.read_csv(
        "../data/training/xgw_feat_importances.csv"
    ).with_columns(pl.lit("ORIGINAL").alias("NETWORK"))
    df_dip_importances = pl.read_csv(
        "../data/training/dip_xgw_feat_importances.csv"
    ).with_columns(pl.lit("DIP").alias("NETWORK"))

    df_importances = pl.concat(
        [df_orig_importances, df_dip_importances], how="vertical"
    )
    df_pandas = (
        df_importances.select(pl.exclude(XVAL_ITER))
        .melt(id_vars="NETWORK", variable_name="FEATURE", value_name="IMPORTANCE")
        .to_pandas()
    )

    plt.figure(dpi=150)
    ax = sns.barplot(
        data=df_pandas, x="FEATURE", y="IMPORTANCE", hue="NETWORK", errorbar=None
    )
    ax.set_title(f"Feature importances on the Original and DIP composite networks.")

    remove_all_axes_labels([ax])
    ax.set_ylabel("IMPORTANCE")
    print(
        df_importances.select(pl.exclude(XVAL_ITER))
        .groupby("NETWORK", maintain_order=True)
        .mean()
    )
    print()


def plot_top_cocomp_edge_pred_losses():
    """
    Prints and plots top 10 methods in terms of co-complex edge classification
    (log loss and Brier score loss).
    """

    print("=====================================================")
    print(
        "TOP 10 METHODS IN TERMS OF CO-COMPLEX EDGE CLASSIFICATION (LOG LOSS & BRIER SCORE LOSS)"
    )
    print()

    df_orig_losses = pl.read_csv("../data/evals/comp_evals.csv").select(
        [METHOD, LOG_LOSS, BRIER_SCORE]
    )
    df_dip_losses = pl.read_csv("../data/evals/dip_comp_evals.csv").select(
        [METHOD, LOG_LOSS, BRIER_SCORE]
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=150)

    def plot(df_losses: pl.DataFrame, ax: Axes, title: str):
        df_losses_avg = (
            df_losses.groupby(METHOD)
            .mean()
            .sort([LOG_LOSS, BRIER_SCORE], descending=[False, False])
            .head(10)
            .melt(id_vars=METHOD, variable_name=METRIC, value_name=VALUE)
        )
        print()
        print(title)
        print(df_losses_avg)
        sns.barplot(
            data=df_losses_avg.to_pandas(), x=METHOD, y=VALUE, hue=METRIC, ax=ax
        )
        ax.set_title(title)
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    fig.suptitle("Classification of co-complex edges (log loss and Brier score loss)")
    plot(df_orig_losses, ax1, "Original Composite Network")
    plot(df_dip_losses, ax2, "DIP Composite Network")

    remove_all_axes_labels([ax1, ax2])
    print()


def plot_top_cocomp_edge_pred_pr_auc():
    """
    Prints and plots top 10 methods in terms of co-complex edge classification (PR AUC).
    """

    print("=====================================================")
    print("TOP 10 METHODS IN TERMS OF CO-COMPLEX EDGE CLASSIFICATION (PR AUC)")
    print()

    df_orig_pr_auc = pl.read_csv("../data/evals/comp_evals.csv").select(
        [METHOD, PR_AUC]
    )
    df_dip_pr_auc = pl.read_csv("../data/evals/dip_comp_evals.csv").select(
        [METHOD, PR_AUC]
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=150)

    def plot(df_pr_auc: pl.DataFrame, ax: Axes, title: str, color: str):
        df_pr_auc_avg = (
            df_pr_auc.groupby(METHOD).mean().sort(PR_AUC, descending=True).head(10)
        )
        print()
        print(title)
        print(df_pr_auc_avg)
        sns.barplot(
            data=df_pr_auc_avg.to_pandas(),
            x=METHOD,
            y=PR_AUC,
            ax=ax,
            color=color,
        )
        ax.set_title(title)
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    fig.suptitle("Classification of co-complex edges (PR AUC)")
    plot(df_orig_pr_auc, ax1, "Original Composite Network", "#4c72b0")
    plot(df_dip_pr_auc, ax2, "DIP Composite Network", "#dd8452")

    remove_all_axes_labels([ax1, ax2])
    ax1.set_ylabel("PR AUC")
    print()


def plot_top_complexes_pred_pr_auc():
    """
    Prints and plots top 8 methods in terms of complex prediction (PR AUC).
    """

    print("=====================================================")
    print("TOP 8 METHODS IN TERMS OF COMPLEX PREDICTION (PR AUC)")
    print()

    df_orig_cluster_evals = pl.read_csv("../data/evals/cluster_evals.csv")
    df_dip_cluster_evals = pl.read_csv("../data/evals/dip_cluster_evals.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=150)

    def plot(df_cluster_evals: pl.DataFrame, ax: Axes, title: str):
        df_evals = prec_recall_curves(df_cluster_evals)
        n_methods = 8
        df_evals_summary = (
            df_evals.groupby(METHOD)
            .mean()
            .sort(AVG_PR_AUC, descending=True)
            .head(n_methods)
            .select(pl.exclude([AVG_PR_AUC, INFLATION]))
            .melt(
                id_vars=METHOD,
                variable_name=SCENARIO,
                value_name=PR_AUC,
            )
        )
        print()
        print(title)
        print(df_evals.groupby(METHOD).mean().sort(AVG_PR_AUC, descending=True))
        sns.barplot(
            data=df_evals_summary.to_pandas(), x=METHOD, y=PR_AUC, hue=SCENARIO, ax=ax
        )
        ax.set_title(title)
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    fig.suptitle("Protein Complex Detection (PR AUC)")
    plot(df_orig_cluster_evals, ax1, "Original Composite Network")
    plot(df_dip_cluster_evals, ax2, "DIP Composite Network")

    remove_all_axes_labels([ax1, ax2])
    ax1.set_ylabel("PR AUC")


def prec_recall_curves(df_cluster_evals: pl.DataFrame):
    """
    Creates four precision-recall curves per inflation setting.
    - all_edges, match_thresh = 0.5
    - all_edges, match_thresh = 0.75
    - 20k_edges, match_thresh = 0.5
    - 20k_edges, match_thresh = 0.75
    """
    df_evals = pl.DataFrame()
    inflations = [2, 3, 4, 5]
    for inflation in inflations:
        df_all_050 = get_prec_recall_curve(
            df_cluster_evals, "all_edges", 0.5, inflation
        )
        df_all_075 = get_prec_recall_curve(
            df_cluster_evals, "all_edges", 0.75, inflation
        )
        df_20k_050 = get_prec_recall_curve(
            df_cluster_evals, "20k_edges", 0.5, inflation
        )
        df_20k_075 = get_prec_recall_curve(
            df_cluster_evals, "20k_edges", 0.75, inflation
        )

        df_all_050_auc = get_prec_recall_auc(df_all_050, "all_050")
        df_all_075_auc = get_prec_recall_auc(df_all_075, "all_075")
        df_20k_050_auc = get_prec_recall_auc(df_20k_050, "20k_050")
        df_20k_075_auc = get_prec_recall_auc(df_20k_075, "20k_075")

        # Print PR_AUC summary of the four scenarios
        df_auc_summary = (
            pl.concat(
                [
                    df_20k_050_auc,
                    df_all_050_auc,
                    df_20k_075_auc,
                    df_all_075_auc,
                ],
                how="vertical",
            )
            .pivot(
                values=PR_AUC,
                index=METHOD,
                columns=SCENARIO,
                aggregate_function="first",
            )
            .with_columns(
                [
                    (pl.sum(pl.all().exclude(METHOD)) / 4).alias(AVG_PR_AUC),
                    pl.lit(inflation).alias(INFLATION),
                ]
            )
        )
        df_evals = pl.concat([df_evals, df_auc_summary], how="vertical")

    return df_evals


def get_prec_recall_curve(
    df_cluster_evals: pl.DataFrame,
    n_edges: str,
    match_thresh: float,
    inflation: int,
) -> pl.DataFrame:
    """
    Gets the precision-recall curve given n_edges, match_thresh, and inflation.
    NOTE: Averages the precision and recall values on all the 10 cross-val iterations.

    Args:
        df_cluster_evals (pl.DataFrame): Cluster evaluations.
        n_edges (str): Either 20k or all_edges.
        match_thresh (float): Match threshold.
        inflation (int): MCL inflation parameter.

    Returns:
        pl.DataFrame: Precision-recall given the supplied settings.
    """
    df_prec_recall = (
        df_cluster_evals.lazy()
        .filter(
            (pl.col(N_EDGES) == n_edges)
            & (pl.col(MATCH_THRESH) == match_thresh)
            & (pl.col(INFLATION) == inflation)
        )
        .groupby([INFLATION, METHOD, DENS_THRESH])
        .mean()
        .groupby([METHOD, DENS_THRESH])
        .mean()
        .sort([METHOD, DENS_THRESH])
        .collect()
    )

    return df_prec_recall


def get_prec_recall_auc(df_prec_recall: pl.DataFrame, scenario: str) -> pl.DataFrame:
    """
    Gets the precision-recall AUC.

    Args:
        df_prec_recall (pl.DataFrame): Precision-recall dataframe.
        scenario (str): Scenario.

    Returns:
        pl.DataFrame: Dataframe containing each method and its PR_AUC.
    """
    df_auc = (
        df_prec_recall.lazy()
        .groupby(METHOD, maintain_order=True)
        .agg(
            pl.struct([PREC, RECALL])
            .apply(
                lambda prec_recall: auc(
                    prec_recall.struct.field(RECALL),
                    prec_recall.struct.field(PREC),
                )
            )
            .alias(PR_AUC)
        )
        .with_columns(pl.lit(scenario).alias(SCENARIO))
        .collect()
    )

    return df_auc


def plot_all_cocomp_edge_pred_losses():
    df_orig_losses = (
        pl.read_csv("../data/evals/comp_evals.csv")
        .select([METHOD, LOG_LOSS, BRIER_SCORE])
        .with_columns(pl.lit("ORIGINAL").alias("NETWORK"))
    )
    df_dip_losses = (
        pl.read_csv("../data/evals/dip_comp_evals.csv")
        .select([METHOD, LOG_LOSS, BRIER_SCORE])
        .with_columns(pl.lit("DIP").alias("NETWORK"))
    )

    df_losses = pl.concat([df_orig_losses, df_dip_losses], how="vertical")
    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=150)

    def plot(df_losses: pl.DataFrame, ax: Axes, metric: str, title: str):
        order = (
            df_losses.select([METHOD, metric])
            .groupby(METHOD)
            .mean()
            .sort(metric, descending=False)
            .select(METHOD)
            .to_series()
            .to_list()
        )
        sns.barplot(
            data=df_losses.to_pandas(),
            x=METHOD,
            y=metric,
            hue="NETWORK",
            ax=ax,
            order=order,
            errorbar=None,
        )
        ax.set_title(title)
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    fig.suptitle(
        "Classification of co-complex edges\n"
        "Loss metrics of all the methods in both the composite networks"
    )
    plot(df_losses, ax1, LOG_LOSS, "Log loss of all the methods")
    plot(df_losses, ax2, BRIER_SCORE, "Brier score loss of all the methods")
    remove_all_axes_labels([ax1, ax2])


def plot_all_cocomp_edge_pred_pr_auc():
    df_orig_pr_auc = (
        pl.read_csv("../data/evals/comp_evals.csv")
        .select([METHOD, PR_AUC])
        .with_columns(pl.lit("ORIGINAL").alias("NETWORK"))
    )
    df_dip_pr_auc = (
        pl.read_csv("../data/evals/dip_comp_evals.csv")
        .select([METHOD, PR_AUC])
        .with_columns(pl.lit("DIP").alias("NETWORK"))
    )

    df_pr_auc = pl.concat([df_orig_pr_auc, df_dip_pr_auc], how="vertical")
    fig, ax = plt.subplots(1, 1, dpi=150)

    def plot(df_pr_auc: pl.DataFrame, ax: Axes, metric: str, title: str):
        order = (
            df_pr_auc.select([METHOD, metric])
            .groupby(METHOD)
            .mean()
            .sort(metric, descending=True)
            .select(METHOD)
            .to_series()
            .to_list()
        )
        sns.barplot(
            data=df_pr_auc.to_pandas(),
            x=METHOD,
            y=metric,
            hue="NETWORK",
            ax=ax,
            order=order,
            errorbar=None,
        )
        ax.set_title(title)
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    fig.suptitle(
        "Classification of co-complex edges\n"
        "PR AUC of all the methods in both the composite networks"
    )
    plot(df_pr_auc, ax, PR_AUC, "")
    remove_all_axes_labels([ax])


def plot_all_complexes_pred_pr_auc():
    df_orig_cluster_evals = pl.read_csv("../data/evals/cluster_evals.csv")
    df_dip_cluster_evals = pl.read_csv("../data/evals/dip_cluster_evals.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=150)

    def plot(df_cluster_evals: pl.DataFrame, ax: Axes, title: str):
        df_evals = prec_recall_curves(df_cluster_evals)
        df_evals_summary = (
            df_evals.groupby(METHOD)
            .mean()
            .sort(AVG_PR_AUC, descending=True)
            .select(pl.exclude([AVG_PR_AUC, INFLATION]))
            .melt(
                id_vars=METHOD,
                variable_name=SCENARIO,
                value_name=PR_AUC,
            )
        )
        sns.barplot(
            data=df_evals_summary.to_pandas(), x=METHOD, y=PR_AUC, hue=SCENARIO, ax=ax
        )
        ax.set_title(title)
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    fig.suptitle(
        "Protein Complex Detection (PR AUC)\n"
        "PR AUC of all the methods in both the composite networks"
    )
    plot(df_orig_cluster_evals, ax1, "Original Composite Network")
    plot(df_dip_cluster_evals, ax2, "DIP Composite Network")

    remove_all_axes_labels([ax1, ax2])


if __name__ == "__main__":
    sns.set_palette("deep")
    pl.Config.set_tbl_rows(20)
    pl.Config.set_tbl_cols(20)

    # # summarizing statistic of the two composite networks
    # summarize_composite_networks()

    # # graphs included in the paper
    # plot_feat_importances()
    # plot_top_cocomp_edge_pred_losses()
    # plot_top_cocomp_edge_pred_pr_auc()
    # plot_top_complexes_pred_pr_auc()

    # supplementary results
    plot_all_cocomp_edge_pred_losses()
    plot_all_cocomp_edge_pred_pr_auc()
    plot_all_complexes_pred_pr_auc()

    plt.show()
