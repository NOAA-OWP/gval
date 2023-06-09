# Get Remote df
import sqlite3 as db
import pandas as pd
import requests
from colorama import Fore, Style


if __name__ == "__main__":
    # Download monitor metrics from testing branch and load in to dataframes
    url = "https://github.com/NOAA-OWP/gval/raw/testing/monitordb"
    req = requests.get(url).content

    with open("a.db", "wb") as file:
        file.write(req)

    remote_con = db.connect("a.db")

    df_metrics_remote = pd.read_sql_query("select * from TEST_METRICS", remote_con)
    df_metrics_remote

    path = "./monitordb"
    local_con = db.connect(path)

    df_metrics_local = pd.read_sql_query("select * from TEST_METRICS", local_con)
    df_sessions = pd.read_sql_query(
        "select * from TEST_SESSIONS ORDER BY RUN_DATE desc", local_con
    )
    session = df_sessions.iloc[0]["SESSION_H"]
    filtered_local = df_metrics_local.loc[df_metrics_local["SESSION_H"] == session]

    merged_df = filtered_local.merge(
        df_metrics_remote,
        on="ITEM_VARIANT",
        how="outer",
        suffixes=("_compare", "_current"),
    )

    # Collect pertinent time benchmarks
    test, perf_change, latest_time, previous_time, difference = [], [], [], [], []
    for xs in merged_df.iterrows():
        x = xs[1]
        test.append(x["ITEM_VARIANT"])

        if x["USER_TIME_compare"] > x["USER_TIME_current"]:
            perf_change.append("Performance Decrease")

        elif x["USER_TIME_compare"] < x["USER_TIME_current"]:
            perf_change.append("Performance Increase")

        else:
            perf_change.append("No Change")

        latest_time.append(x["USER_TIME_compare"])
        previous_time.append(x["USER_TIME_current"])
        difference.append(x["USER_TIME_compare"] - x["USER_TIME_current"])

    final_df = pd.DataFrame(
        {
            "Test": test,
            "Performance Status": perf_change,
            "Latest Time (s)": latest_time,
            "Previous Time (s)": previous_time,
            "Difference (s)": difference,
        }
    )

    # Save report and display results in terminal
    final_df.to_csv("./remote_performance_report.csv")
    final_df["test_len"] = final_df["Test"].apply(lambda x: len(x.split("[")[0]))
    max_len = final_df["test_len"].max()

    print(
        " Test" + " " * (max_len - 4),
        "    Latest Time (s)",
        "      Previous Time (s)",
        "     Difference (s)",
    )
    for xs in final_df.iterrows():
        x = xs[1]

        if x["Difference (s)"] < 0:
            color = Fore.GREEN
        elif x["Difference (s)"] > 0:
            color = Fore.RED
        else:
            color = Style.RESET_ALL

        str_latest_time = str(x["Latest Time (s)"])[:8]
        str_prev_time = str(x["Previous Time (s)"])[:8]
        str_diff_time = str(x["Difference (s)"])[:8]

        spaces = max_len - len(x["Test"].split("[")[0])
        latest_time_spaces = (
            0 if len(str_latest_time) >= 8 else 8 - len(str_latest_time)
        )
        prev_time_spaces = 0 if len(str_prev_time) >= 8 else 8 - len(str_prev_time)
        difference_time_spaces = (
            0 if len(str_diff_time) >= 8 else 8 - len(str_diff_time)
        )

        print(
            color,
            f"{x['Test'].split('[')[0]}" + " " * spaces,
            f"     -     {str_latest_time}" + " " * latest_time_spaces,
            f"     -     {str_prev_time}" + " " * prev_time_spaces,
            f"     -     {str_diff_time}" + " " * difference_time_spaces,
        )

    total_perf = final_df["Difference (s)"].sum()
    description = (
        "Total loss in CPU TIME performance by"
        if total_perf > 0
        else "Total CPU TIME performance gain of "
    )
    print(Style.RESET_ALL, "\n", f"{description} {abs(total_perf)} seconds")
