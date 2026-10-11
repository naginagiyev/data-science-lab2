import re
import warnings
import numpy as np
import pandas as pd
from typing import Any
from pathlib import Path
from tabulate import tabulate
from datetime import timedelta
from itertools import combinations
from tools.datareader import load_data
from configuration.edaconfig import allowed_task_types
from scipy.stats import chi2_contingency as chi2_contingency, kurtosis, pointbiserialr, skew, zscore

class EDA:
    # initialze the class
    def __init__(self, input_path: str, target_col: str, task_type: str):
        self.input_path = input_path

        # get the target column
        self.target_col = str(target_col).strip() if target_col and str(target_col).strip() else None
        if task_type not in allowed_task_types:
            raise ValueError(
                "taskType must be one of: "
                "binary-classification, regression, multi-class-classification."
            )
        self.task_type = task_type

        # create the output path
        input_obj = Path(input_path)
        self.output_path = str(input_obj.with_name(f"{input_obj.stem}eda.md"))

        # initialize the dataframe
        self.df: pd.DataFrame | None = None
        # initialize the column types
        self.col_types: dict[str, str] = {}
        # initialize the warnings
        self.warnings: list[str] = []
        # initialize the report sections
        self.report_sections: list[str] = []
        # initialize the outlier statistics
        self.outlier_stats: dict[str, dict[str, float]] = {}

    # the sections that will be run and written to the report
    def run(self) -> str:
        # load the data and get the types
        self.df = load_data(self.input_path, self.target_col)
        self.infer_types()

        # main sections of the report
        self.report_sections.append(self.section_overview())
        self.report_sections.append(self.section_target())
        self.report_sections.append(self.section_columns())
        self.report_sections.append(self.section_missing())
        self.report_sections.append(self.section_correlations())
        self.report_sections.append(self.section_outliers())
        self.report_sections.append(self.section_warnings())

        # write the report into an .md file
        self.write_report()
        return self.output_path

    # a function to decide what is the real type of the columns in the dataset
    def infer_types(self) -> None:
        if self.df is None:
            raise ValueError("Data is not loaded.")

        df = self.df
        n_rows = len(df)

        # iterate through the columns and decide the type
        for col in df.columns:
            series = df[col]
            non_null = series.dropna()
            non_null_count = len(non_null)
            unique_count = non_null.nunique(dropna=True)
            unique_ratio = (unique_count / n_rows) if n_rows else 0.0

            if non_null_count == 0:
                self.col_types[col] = "categorical"
                continue

            if n_rows >= 200:
                if unique_ratio >= 0.98:
                    self.col_types[col] = "id_like"
                    continue
            else:
                if unique_count == non_null_count and non_null_count >= 10:
                    self.col_types[col] = "id_like"
                    continue

            converted_numeric = self.try_convert_string_numeric(col)
            if converted_numeric:
                series = self.df[col]
                non_null = series.dropna()
                non_null_count = len(non_null)
                unique_count = non_null.nunique(dropna=True)
                unique_ratio = (unique_count / n_rows) if n_rows else 0.0

            if self.is_boolean_series(non_null):
                self.col_types[col] = "boolean"
                continue

            if pd.api.types.is_datetime64_any_dtype(series):
                self.col_types[col] = "datetime"
                continue

            if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                if self.datetime_parse_success_ratio(non_null) > 0.80:
                    self.col_types[col] = "datetime"
                    continue
                avg_words = self.average_word_count(non_null)
                if n_rows >= 200:
                    if avg_words > 3 and unique_ratio >= 0.40:
                        self.col_types[col] = "text"
                        continue
                else:
                    if avg_words > 3 and unique_count >= 20:
                        self.col_types[col] = "text"
                        continue
                if non_null.map(type).nunique() > 1:
                    self.add_warning(f"Mixed types detected in column '{col}'. Treated as categorical.")
                self.col_types[col] = "categorical"
                continue

            if pd.api.types.is_numeric_dtype(series):
                if n_rows >= 200:
                    is_discrete = unique_ratio <= 0.05
                else:
                    is_discrete = unique_count <= 15
                if is_discrete:
                    self.col_types[col] = "numeric_discrete"
                else:
                    self.col_types[col] = "numeric_continuous"
                continue

            self.col_types[col] = "categorical"

    # overview section and columns summary table
    def section_overview(self) -> str:
        if self.df is None:
            return "## 1. Dataset Overview\n\nNo data loaded.\n"

        df = self.df
        inp = Path(self.input_path)
        file_size = inp.stat().st_size if inp.exists() else 0
        memory_usage = int(df.memory_usage(deep=True).sum())
        missing_count = int(df.isna().sum().sum())
        total_cells = int(df.shape[0] * df.shape[1])
        missing_pct = self.percent(missing_count, total_cells)

        duplicate_mask = df.duplicated()
        duplicate_count = int(duplicate_mask.sum())
        duplicate_pct = self.percent(duplicate_count, len(df))
        duplicate_indices = df.index[duplicate_mask].tolist()[:3]

        type_breakdown = pd.Series(self.col_types).value_counts().to_dict()
        type_line = ", ".join(f"{v} {k}" for k, v in type_breakdown.items()) if type_breakdown else "N/A"

        summary_rows: list[list[Any]] = []
        for col in df.columns:
            miss = int(df[col].isna().sum())
            summary_rows.append(
                [
                    col,
                    self.col_types.get(col, "unknown"),
                    str(df[col].dtype),
                    miss,
                    f"{self.percent(miss, len(df)):.2f}%",
                ]
            )

        section = [
            "## 1. Dataset Overview",
            f"- File name: `{Path(self.input_path).name}`",
            f"- File size: {self.format_bytes(file_size)}",
            f"- Rows: {len(df)}",
            f"- Columns: {df.shape[1]}",
            f"- Memory usage: {self.format_bytes(memory_usage)}",
            f"- Missing values: {missing_count} ({missing_pct:.2f}%)",
            f"- Duplicate rows: {duplicate_count} ({duplicate_pct:.2f}%)",
            f"- First duplicate row indices (up to 3): {duplicate_indices if duplicate_indices else 'None'}",
            f"- Column type breakdown: {type_line}",
            "",
            "### Column Summary",
            self.md_table(
                summary_rows,
                ["Column", "Inferred Type", "Pandas Dtype", "Missing Count", "Missing %"],
            ),
            "",
        ]
        return "\n".join(section)

    # target column analysis
    def section_target(self) -> str:
        # make sure the dataframe is loaded
        if self.df is None:
            return ""

        target = self.df[self.target_col]
        task_type = self.task_type
        non_null = target.dropna()
        missing_count = int(target.isna().sum())
        missing_pct = self.percent(missing_count, len(target))

        section = ["## 2. Target Column Analysis", f"- Target column: `{self.target_col}`", f"- Task type: `{task_type}`"]

        # make sure the target column has more than one unique value
        if non_null.nunique(dropna=True) <= 1:
            self.add_warning("Target column has only one unique non-null value.")

        # check the imbalance of the target column
        if task_type in {"binary-classification", "multi-class-classification"}:
            counts = non_null.value_counts(dropna=True)
            total = int(counts.sum())
            class_rows = [[idx, int(val), f"{self.percent(int(val), total):.2f}%"] for idx, val in counts.items()]

            if len(counts) > 0:
                min_count = int(counts.min())
                max_count = int(counts.max())
                minority_ratio = (min_count / total) if total else 0.0
                imbalance_ratio = (max_count / min_count) if min_count else np.inf
                imbalance_level = self.class_imbalance_level(imbalance_ratio)

                section.extend(
                    [
                        f"- Class imbalance level: {imbalance_level}",
                        f"- Minority class share: {minority_ratio * 100:.2f}%",
                        f"- Imbalance ratio (majority/minority): {imbalance_ratio:.2f}",
                    ]
                )

                if imbalance_level in {"moderate imbalance", "strong imbalance", "very strong imbalance"}:
                    self.add_warning(
                        f"Class imbalance in target '{self.target_col}': {imbalance_level} "
                        f"(minority class is {minority_ratio * 100:.2f}%, ratio={imbalance_ratio:.2f})."
                    )

            section.extend(
                [
                    "",
                    "### Class Distribution",
                    self.md_table(class_rows, ["Class", "Count", "Percentage"]),
                    f"- Missing values in target: {missing_count} ({missing_pct:.2f}%)",
                ]
            )
        elif task_type == "regression":
            numeric_target = pd.to_numeric(non_null, errors="coerce").dropna()
            q1 = float(numeric_target.quantile(0.25)) if len(numeric_target) else 0.0
            q3 = float(numeric_target.quantile(0.75)) if len(numeric_target) else 0.0
            iqr = q3 - q1
            lower = q1 - (1.5 * iqr)
            upper = q3 + (1.5 * iqr)
            outlier_count = int(((numeric_target < lower) | (numeric_target > upper)).sum())
            stats_rows = [
                ["count", float(numeric_target.count())],
                ["mean", float(numeric_target.mean()) if len(numeric_target) else np.nan],
                ["median", float(numeric_target.median()) if len(numeric_target) else np.nan],
                ["std", float(numeric_target.std()) if len(numeric_target) > 1 else np.nan],
                ["min", float(numeric_target.min()) if len(numeric_target) else np.nan],
                ["max", float(numeric_target.max()) if len(numeric_target) else np.nan],
                ["skewness", float(skew(numeric_target, nan_policy="omit")) if len(numeric_target) > 2 else np.nan],
                ["kurtosis", float(kurtosis(numeric_target, nan_policy="omit")) if len(numeric_target) > 3 else np.nan],
            ]
            pct_rows = [
                ["5th", float(numeric_target.quantile(0.05)) if len(numeric_target) else np.nan],
                ["25th", q1],
                ["75th", q3],
                ["95th", float(numeric_target.quantile(0.95)) if len(numeric_target) else np.nan],
            ]
            section.extend(
                [
                    "",
                    "### Target Statistics",
                    self.md_table(stats_rows, ["Metric", "Value"]),
                    "",
                    "### Target Percentiles",
                    self.md_table(pct_rows, ["Percentile", "Value"]),
                    f"- IQR outlier count: {outlier_count}",
                    f"- Missing values in target: {missing_count} ({missing_pct:.2f}%)",
                ]
            )
        else:
            section.append("- Target analysis summary is limited for this task type.")
            section.append(f"- Missing values in target: {missing_count} ({missing_pct:.2f}%)")

        section.append("")
        return "\n".join(section)

    # individual column analysis
    def section_columns(self) -> str:
        if self.df is None:
            return "## 3. Per-Column Analysis\n\nNo data loaded.\n"

        # iterate over all columns and analyze them
        section_parts = ["## 3. Per-Column Analysis"]
        for col in self.df.columns:
            col_type = self.col_types.get(col, "categorical")
            if col_type in {"numeric_continuous", "numeric_discrete"}:
                section_parts.append(self.analyze_numeric(col))
            elif col_type == "categorical":
                section_parts.append(self.analyze_categorical(col))
            elif col_type == "boolean":
                section_parts.append(self.analyze_boolean(col))
            elif col_type == "datetime":
                section_parts.append(self.analyze_datetime(col))
            elif col_type == "text":
                section_parts.append(self.analyze_text(col))
            elif col_type == "id_like":
                section_parts.append(self.analyze_id(col))
            else:
                section_parts.append(self.analyze_categorical(col))

        return "\n\n".join(section_parts)

    # function to specifically analyze numeric columns
    def analyze_numeric(self, col: str) -> str:
        # make sure the dataframe is loaded
        if self.df is None:
            return f"### {col}\n\nNo data loaded."

        raw_series = self.df[col]
        series = pd.to_numeric(raw_series, errors="coerce")
        non_null = series.dropna()
        total_rows = len(raw_series)
        non_null_count = int(non_null.count())
        missing_count = int(raw_series.isna().sum())
        missing_pct = self.percent(missing_count, total_rows)
        unique_count = int(non_null.nunique(dropna=True))

        # make sure not all values are missing
        if non_null_count == 0:
            self.add_warning(f"Column '{col}' is 100% missing and numeric statistics were skipped.")
            return (
                f"### {col}\n"
                f"- Inferred type: `{self.col_types.get(col, 'unknown')}`\n"
                f"- Pandas dtype: `{raw_series.dtype}`\n"
                f"- Missing: {missing_count} ({missing_pct:.2f}%)\n"
                "- All values are missing.\n"
            )

        # mode of the column
        mode_series = non_null.mode(dropna=True)
        mode_value = mode_series.iloc[0] if not mode_series.empty else np.nan

        # quantiles of the column
        q1 = float(non_null.quantile(0.25))
        q3 = float(non_null.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        # IQR outliers of the column
        iqr_mask = (non_null < lower) | (non_null > upper)
        iqr_outliers = int(iqr_mask.sum())
        iqr_pct = self.percent(iqr_outliers, non_null_count)

        # Z-score outliers of the column
        if non_null_count > 2:
            zscores = pd.Series(zscore(non_null, nan_policy="omit"), index=non_null.index)
            z_outliers = int((zscores.abs() > 3).sum())
        else:
            z_outliers = 0
        z_pct = self.percent(z_outliers, non_null_count)

        # skew and kurtosis of the column
        skew_val = float(skew(non_null, nan_policy="omit")) if non_null_count > 2 else np.nan
        kurt_val = float(kurtosis(non_null, nan_policy="omit")) if non_null_count > 3 else np.nan

        if not np.isnan(skew_val):
            if abs(skew_val) > 2:
                self.add_warning(f"Severe skew in '{col}' (skew={skew_val:.4f}).")
            elif abs(skew_val) > 1:
                self.add_warning(f"High skew in '{col}' (skew={skew_val:.4f}).")
        if iqr_pct > 5:
            self.add_warning(f"Outliers above 5% in '{col}' by IQR method ({iqr_pct:.2f}%).")

        self.outlier_stats[col] = {
            "iqr_count": float(iqr_outliers),
            "iqrPct": float(iqr_pct),
            "z_count": float(z_outliers),
            "zPct": float(z_pct),
        }

        percentiles = [0.01, 0.05, 0.25, 0.75, 0.95, 0.99]
        percentile_rows = [[f"{int(p * 100)}th", float(non_null.quantile(p))] for p in percentiles]

        zero_count = int((non_null == 0).sum())
        zero_pct = self.percent(zero_count, non_null_count)
        negative_count = int((non_null < 0).sum())
        negative_pct = self.percent(negative_count, non_null_count)

        lines = [
            f"### {col}",
            f"- Inferred type: `{self.col_types.get(col, 'unknown')}`",
            f"- Pandas dtype: `{raw_series.dtype}`",
            f"- Non-null count: {non_null_count}",
            f"- Missing: {missing_count} ({missing_pct:.2f}%)",
            f"- Unique values: {unique_count}",
            f"- Min: {float(non_null.min()):.4f}",
            f"- Max: {float(non_null.max()):.4f}",
            f"- Mean: {float(non_null.mean()):.4f}",
            f"- Median: {float(non_null.median()):.4f}",
            f"- Mode: {float(mode_value):.4f}" if pd.notna(mode_value) else "- Mode: N/A",
            f"- Standard deviation: {float(non_null.std()):.4f}" if non_null_count > 1 else "- Standard deviation: N/A",
            f"- Skewness: {skew_val:.4f}" if not np.isnan(skew_val) else "- Skewness: N/A",
            f"- Kurtosis: {kurt_val:.4f}" if not np.isnan(kurt_val) else "- Kurtosis: N/A",
            "",
            "#### Percentiles",
            self.md_table(percentile_rows, ["Percentile", "Value"]),
            f"- Zero count: {zero_count} ({zero_pct:.2f}%)",
        ]

        # negative values of the column
        if negative_count > 0:
            lines.append(f"- Negative count: {negative_count} ({negative_pct:.2f}%)")

        # value counts of the column
        if self.col_types.get(col) == "numeric_discrete" and unique_count <= 20:
            value_counts = (
                non_null.value_counts(dropna=True)
                .sort_index()
                .rename_axis("Value")
                .reset_index(name="Count")
            )
            rows = [
                [row["Value"], int(row["Count"]), f"{self.percent(int(row['Count']), non_null_count):.2f}%"]
                for _, row in value_counts.iterrows()
            ]
            lines.extend(["", "#### Value Counts", self.md_table(rows, ["Value", "Count", "%"])])

        return "\n".join(lines)

    # function to specifically analyze categorical columns
    def analyze_categorical(self, col: str) -> str:
        # make sure the dataframe is loaded
        if self.df is None:
            return f"### {col}\n\nNo data loaded."

        # get some statistics about the categorical column  like 
        # non-null count, missing count, unique count, mode, least frequent value, top rows
        series = self.df[col]
        non_null = series.dropna().astype(str)
        total_rows = len(series)
        non_null_count = int(non_null.count())
        missing_count = int(series.isna().sum())
        missing_pct = self.percent(missing_count, total_rows)
        unique_count = int(non_null.nunique(dropna=True))

        # check if the unique count is greater than 50
        if unique_count > 50:
            self.add_warning(f"High cardinality in '{col}' ({unique_count} unique values).")
        # check if the unique count is equal to the total rows
        if unique_count == total_rows and total_rows > 0:
            self.add_warning(f"Column '{col}' appears identifier-like (cardinality equals rows).")

        vc_desc = non_null.value_counts(dropna=True)
        vc_asc = non_null.value_counts(dropna=True, ascending=True)
        mode_value = vc_desc.index[0] if not vc_desc.empty else "N/A"
        least_value = vc_asc.index[0] if not vc_asc.empty else "N/A"
        least_count = int(vc_asc.iloc[0]) if not vc_asc.empty else 0

        top10 = vc_desc.head(10)
        top_rows = [[idx, int(cnt), f"{self.percent(int(cnt), non_null_count):.2f}%"] for idx, cnt in top10.items()]

        lines = [
            f"### {col}",
            f"- Inferred type: `{self.col_types.get(col, 'unknown')}`",
            f"- Pandas dtype: `{series.dtype}`",
            f"- Non-null count: {non_null_count}",
            f"- Missing: {missing_count} ({missing_pct:.2f}%)",
            f"- Cardinality (unique): {unique_count}",
            f"- Mode: `{mode_value}`",
            f"- Least frequent value: `{least_value}` ({least_count})",
            "",
            "#### Top 10 Value Counts",
            self.md_table(top_rows, ["Value", "Count", "Percentage"]) if top_rows else "No non-null values.",
        ]

        if unique_count > 10:
            lines.append(f"- Showing top 10 of {unique_count} unique values.")

        return "\n".join(lines)

    # function to specifically analyze boolean columns
    def analyze_boolean(self, col: str) -> str:
        # make sure the dataframe is loaded
        if self.df is None:
            return f"### {col}\n\nNo data loaded."

        # get some statistics about the boolean column like
        # non-null count, missing count, value counts, true ratio
        # also normalize the boolean values to 1 and 0
        series = self.df[col]
        non_null = series.dropna()
        missing_count = int(series.isna().sum())
        missing_pct = self.percent(missing_count, len(series))

        normalized = non_null.map(self.normalize_bool_token)
        vc = non_null.astype(str).value_counts(dropna=True)
        total_nonNull = int(non_null.count())

        true_ratio = self.percent(int((normalized == 1).sum()), total_nonNull) / 100 if total_nonNull else 0.0
        rows = [[idx, int(cnt), f"{self.percent(int(cnt), total_nonNull):.2f}%"] for idx, cnt in vc.items()]

        detected_values = sorted(non_null.astype(str).unique().tolist())

        lines = [
            f"### {col}",
            f"- Inferred type: `{self.col_types.get(col, 'unknown')}`",
            f"- Detected boolean-like values: {detected_values}",
            f"- Missing: {missing_count} ({missing_pct:.2f}%)",
            "",
            "#### Value Counts",
            self.md_table(rows, ["Value", "Count", "Percentage"]) if rows else "No non-null values.",
            f"- True ratio: {true_ratio:.4f}",
        ]
        return "\n".join(lines)

    # function to specifically analyze datetime columns
    def analyze_datetime(self, col: str) -> str:
        # make sure the dataframe is loaded
        if self.df is None:
            return f"### {col}\n\nNo data loaded."

        # get some statistics about the datetime column like
        # non-null count, missing count, earliest date, latest date, total range (days),
        # is monotonically increasing, min gap, max gap, mean gap
        series = self.df[col]
        parsed = pd.to_datetime(series, errors="coerce")
        non_null = parsed.dropna()
        missing_count = int(parsed.isna().sum())
        missing_pct = self.percent(missing_count, len(parsed))

        if non_null.empty:
            return (
                f"### {col}\n"
                f"- Inferred type: `{self.col_types.get(col, 'unknown')}`\n"
                f"- Missing: {missing_count} ({missing_pct:.2f}%)\n"
                "- No parseable datetime values found.\n"
            )

        earliest = non_null.min()
        latest = non_null.max()
        date_range_days = int((latest - earliest).days)
        monotonic = non_null.is_monotonic_increasing

        years = non_null.dt.year.value_counts().head(3)
        months = non_null.dt.month_name().value_counts().head(3)
        weekdays = non_null.dt.day_name().value_counts().head(3)

        sorted_vals = non_null.sort_values()
        gaps = sorted_vals.diff().dropna()
        min_gap = self.format_timedelta(gaps.min()) if not gaps.empty else "N/A"
        max_gap = self.format_timedelta(gaps.max()) if not gaps.empty else "N/A"
        mean_gap = self.format_timedelta(gaps.mean()) if not gaps.empty else "N/A"

        lines = [
            f"### {col}",
            f"- Inferred type: `{self.col_types.get(col, 'unknown')}`",
            f"- Missing: {missing_count} ({missing_pct:.2f}%)",
            f"- Earliest date: {earliest}",
            f"- Latest date: {latest}",
            f"- Total range (days): {date_range_days}",
            f"- Is monotonically increasing: {'yes' if monotonic else 'no'}",
            f"- Min gap: {min_gap}",
            f"- Max gap: {max_gap}",
            f"- Mean gap: {mean_gap}",
            f"- Top years: {', '.join(f'{k} ({v})' for k, v in years.items())}",
            f"- Top months: {', '.join(f'{k} ({v})' for k, v in months.items())}",
            f"- Top weekdays: {', '.join(f'{k} ({v})' for k, v in weekdays.items())}",
        ]
        return "\n".join(lines)

    # function to specifically analyze text columns
    def analyze_text(self, col: str) -> str:
        # make sure the dataframe is loaded
        if self.df is None:
            return f"### {col}\n\nNo data loaded."

        # get some statistics about the text column like
        # non-null count, missing count, character length, word length, empty string count
        series = self.df[col]
        text = series.dropna().astype(str)
        stripped = text.str.strip()
        missing_count = int(series.isna().sum())
        missing_pct = self.percent(missing_count, len(series))

        char_lens = stripped.str.len() if not stripped.empty else pd.Series(dtype=float)
        word_lens = stripped.str.split().str.len() if not stripped.empty else pd.Series(dtype=float)
        empty_count = int((stripped == "").sum()) if not stripped.empty else 0

        lines = [
            f"### {col}",
            "- Flag: text column",
            f"- Non-null count: {int(text.count())}",
            f"- Missing: {missing_count} ({missing_pct:.2f}%)",
            f"- Character length (min/max/mean): {self.series_min_max_mean(char_lens)}",
            f"- Word count (min/max/mean): {self.series_min_max_mean(word_lens)}",
            f"- Empty string count: {empty_count}",
            "- Note: Deep NLP analysis not included.",
        ]
        return "\n".join(lines)

    # function to specifically analyze identifier columns
    def analyze_id(self, col: str) -> str:
        # make sure the dataframe is loaded
        if self.df is None:
            return f"### {col}\n\nNo data loaded."

        # get some statistics about the identifier column like
        # unique count, unique ratio, sample values
        series = self.df[col]
        unique_count = int(series.nunique(dropna=True))
        unique_ratio = self.percent(unique_count, len(series))
        sample_values = series.dropna().astype(str).head(3).tolist()

        lines = [
            f"### {col}",
            "- Flag: Likely an identifier column - deep analysis skipped.",
            f"- Unique count: {unique_count}",
            f"- Unique ratio: {unique_ratio:.2f}%",
            f"- Sample values (up to 3): {sample_values if sample_values else 'N/A'}",
        ]
        return "\n".join(lines)

    # SECTION 4 - Missing values
    # here we will analyze the missing values in the dataframe
    # we will check for total missing cells, columns with missing values, columns with 0% missing,
    # columns with >50% missing, columns with 100% missing, and co-missing pairs (>50%)
    # we will also add warnings for columns with high missingness, fully missing columns, and co-missing pairs (>50%)
    def section_missing(self) -> str:
        if self.df is None:
            return "## 4. Missing Values\n\nNo data loaded.\n"

        df = self.df
        total_missing = int(df.isna().sum().sum())
        section = ["## 4. Missing Values", f"- Total missing cells: {total_missing}"]

        missing_cols = []
        no_missingCols = []

        for col in df.columns:
            miss = int(df[col].isna().sum())
            miss_pct = self.percent(miss, len(df))
            if miss > 0:
                missing_cols.append([col, miss, f"{miss_pct:.2f}%", self.col_types.get(col, "unknown"), miss_pct])
            else:
                no_missingCols.append(col)

            if miss_pct > 50:
                self.add_warning(f"Column '{col}' has high missingness ({miss_pct:.2f}%).")
            if miss_pct == 100:
                self.add_warning(f"Column '{col}' is fully missing (100%).")

        if missing_cols:
            missing_cols_sorted = sorted(missing_cols, key=lambda row: row[4], reverse=True)
            display_rows = [row[:4] for row in missing_cols_sorted]
            section.extend(
                [
                    "",
                    "### Columns With Missing Values",
                    self.md_table(display_rows, ["Column", "Missing Count", "Missing %", "Inferred Type"]),
                ]
            )
        else:
            section.append("- No missing values detected in any column.")

        section.append(
            f"- Columns with 0% missing: {len(no_missingCols)} of {len(df.columns)}"
        )

        missing_columns = [col for col in df.columns if df[col].isna().sum() > 0]
        co_missing_rows: list[list[Any]] = []
        for col_a, col_b in combinations(missing_columns, 2):
            both_missing = int((df[col_a].isna() & df[col_b].isna()).sum())
            pct = self.percent(both_missing, len(df))
            if pct > 50:
                co_missing_rows.append([col_a, col_b, both_missing, f"{pct:.2f}%"])

        if co_missing_rows:
            section.extend(
                [
                    "",
                    "### Co-missing Pairs (>50%)",
                    self.md_table(
                        co_missing_rows,
                        ["Column A", "Column B", "Co-missing Rows", "Co-missing %"],
                    ),
                ]
            )
        else:
            section.append("- No high co-missing pairs (>50%) found.")

        section.append("")
        return "\n".join(section)

    # SECTION 5 - Correlations
    # here we will analyze the correlations between the columns in the dataframe
    # we will check for numeric vs numeric correlations, categorical vs categorical correlations, and feature vs target correlations
    # we will also add warnings for potential multicollinearity, and no strong numeric correlations found
    def section_correlations(self) -> str:
        if self.df is None:
            return "## 5. Correlations\n\nNo data loaded.\n"

        df = self.df
        section = ["## 5. Correlations", "### Numeric vs Numeric (Pearson)"]

        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        if len(numeric_cols) < 2:
            section.append("- Not enough numeric columns for Pearson correlation.")
        else:
            corr = df[numeric_cols].corr(numeric_only=True)
            pairs = []
            for i, col_a in enumerate(corr.columns):
                for col_b in corr.columns[i + 1 :]:
                    val = corr.loc[col_a, col_b]
                    if pd.notna(val) and abs(val) > 0.5:
                        strength = self.corr_strength(abs(float(val)))
                        pairs.append([col_a, col_b, float(val), strength, abs(float(val))])
                        if abs(val) > 0.9:
                            self.add_warning(
                                f"Potential multicollinearity: '{col_a}' vs '{col_b}' (r={val:.4f})."
                            )
            if pairs:
                pairs = sorted(pairs, key=lambda row: row[4], reverse=True)
                table_rows = [[a, b, r, s] for a, b, r, s, _ in pairs]
                section.append(self.md_table(table_rows, ["Feature A", "Feature B", "Pearson r", "Strength"]))
            else:
                section.append("- No strong numeric correlations found.")

        section.extend(["", "### Categorical vs Categorical (Cramer's V)"])
        cat_cols = [col for col in df.columns if self.col_types.get(col) == "categorical"]
        cramers_rows = []
        for col_a, col_b in combinations(cat_cols, 2):
            if df[col_a].nunique(dropna=True) > 50 or df[col_b].nunique(dropna=True) > 50:
                continue
            v = self.cramers_v(df[col_a], df[col_b])
            if v > 0.3:
                strength = self.corr_strength(v, categorical=True)
                cramers_rows.append([col_a, col_b, v, strength, v])

        if cramers_rows:
            cramers_rows = sorted(cramers_rows, key=lambda row: row[4], reverse=True)
            section.append(
                self.md_table(
                    [[a, b, v, s] for a, b, v, s, _ in cramers_rows],
                    ["Feature A", "Feature B", "Cramer's V", "Strength"],
                )
            )
        else:
            section.append("- No categorical pairs exceeded Cramer's V > 0.3.")

        # find correlations between features and target
        # if the target column is a regression target, we will check for Pearson r correlations
        # if the target column is a classification target, we will check for Cramer's V and point-biserial correlations
        # if the target column is a binary classification target, we will check for point-biserial correlations
        if self.target_col in df.columns:
            section.extend(["", "### Feature vs Target"])
            target_type = self.task_type
            target = df[self.target_col]
            feature_rows = []

            if target_type == "regression":
                target_num = pd.to_numeric(target, errors="coerce")
                for col in numeric_cols:
                    if col == self.target_col:
                        continue
                    pair_df = pd.DataFrame({"x": pd.to_numeric(df[col], errors="coerce"), "y": target_num}).dropna()
                    if len(pair_df) < 2:
                        continue
                    val = pair_df["x"].corr(pair_df["y"])
                    if pd.notna(val) and abs(val) > 0.1:
                        feature_rows.append([col, "Pearson r", float(val), abs(float(val))])
            else:
                target_nonNull = target.dropna()
                target_unique = target_nonNull.nunique(dropna=True)
                for col in cat_cols:
                    if col == self.target_col:
                        continue
                    v = self.cramers_v(df[col], target)
                    if pd.notna(v) and v > 0.1:
                        feature_rows.append([col, "Cramer's V", float(v), abs(float(v))])

                if target_unique == 2:
                    valid_target = target_nonNull.astype("category")
                    mapping = {cat: idx for idx, cat in enumerate(valid_target.cat.categories)}
                    encoded_target = target.map(mapping)
                    for col in numeric_cols:
                        if col == self.target_col:
                            continue
                        pair_df = pd.DataFrame(
                            {"x": pd.to_numeric(df[col], errors="coerce"), "y": encoded_target}
                        ).dropna()
                        if len(pair_df) < 2:
                            continue
                        stat, _ = pointbiserialr(pair_df["y"], pair_df["x"])
                        if pd.notna(stat) and abs(stat) > 0.1:
                            feature_rows.append([col, "Point-Biserial", float(stat), abs(float(stat))])
                else:
                    section.append("- Point-biserial is skipped because target is not binary.")

            if feature_rows:
                feature_rows = sorted(feature_rows, key=lambda row: row[3], reverse=True)
                section.append(
                    self.md_table(
                        [[a, b, c] for a, b, c, _ in feature_rows],
                        ["Feature", "Metric", "Value"],
                    )
                )
            else:
                section.append("- No valid feature-target relationships were computed.")

        section.append("")
        return "\n".join(section)

    # SECTION 6 - Outliers
    # here we will analyze the outliers in the dataframe
    # we will check for IQR outliers and Z-score outliers in the numeric columns
    # we will also add warnings for columns with >5% IQR outliers
    def section_outliers(self) -> str:
        section = ["## 6. Outlier Summary"]
        if not self.outlier_stats:
            section.extend(["- No numeric columns found for outlier analysis.", ""])
            return "\n".join(section)

        rows = []
        for col, stats in self.outlier_stats.items():
            rows.append(
                [
                    col,
                    int(stats["iqr_count"]),
                    f"{stats['iqrPct']:.2f}%",
                    int(stats["z_count"]),
                    f"{stats['zPct']:.2f}%",
                    stats["iqr_count"],
                ]
            )
            if stats["iqrPct"] > 5:
                self.add_warning(f"Outlier summary flags '{col}' above 5% IQR outliers.")

        rows_sorted = sorted(rows, key=lambda row: row[5], reverse=True)
        display = [row[:5] for row in rows_sorted]
        section.append(
            self.md_table(
                display,
                ["Column", "IQR Outlier Count", "IQR Outlier %", "Z-score Outlier Count", "Z-score Outlier %"],
            )
        )
        section.append("")
        return "\n".join(section)

    # SECTION 7 - Warnings
    # here is the section that contains warnings we added along our EDA journey
    # we will group the warnings by category and add them to the section
    def section_warnings(self) -> str:
        section = ["## 7. Data Quality Warnings"]
        if not self.warnings:
            section.extend(["No data quality issues detected.", ""])
            return "\n".join(section)

        grouped: dict[str, list[str]] = {
            "Missing Data": [],
            "Outliers": [],
            "Correlations": [],
            "Cardinality": [],
            "Skewness": [],
            "Class Imbalance": [],
            "Other": [],
        }

        for msg in self.warnings:
            lowered = msg.lower()
            if "missing" in lowered:
                grouped["Missing Data"].append(msg)
            elif "outlier" in lowered:
                grouped["Outliers"].append(msg)
            elif "correlation" in lowered or "multicollinearity" in lowered:
                grouped["Correlations"].append(msg)
            elif "cardinality" in lowered or "identifier-like" in lowered:
                grouped["Cardinality"].append(msg)
            elif "skew" in lowered:
                grouped["Skewness"].append(msg)
            elif "imbalance" in lowered or "minority class" in lowered:
                grouped["Class Imbalance"].append(msg)
            else:
                grouped["Other"].append(msg)

        idx = 1
        for group_name, items in grouped.items():
            if not items:
                continue
            section.append(f"### {group_name}")
            for item in items:
                section.append(f"{idx}. {item}")
                idx += 1
            section.append("")

        return "\n".join(section)

    # simple function to add warnings to the warnings list
    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    # function to convert a list of lists to a markdown table
    def md_table(self, data: list[list[Any]], headers: list[str]) -> str:
        return tabulate(data, headers=headers, tablefmt="pipe", floatfmt=".4f")

    # function to calculate the percentage safely
    @staticmethod
    def percent(part: float, whole: float) -> float:
        if whole == 0:
            return 0.0
        return round((part / whole) * 100, 2)

    # function to calculate the class imbalance level
    @staticmethod
    def class_imbalance_level(imbalance_ratio: float) -> str:
        if imbalance_ratio <= 1.5:
            return "no imbalance"
        if imbalance_ratio <= 2.0:
            return "weak imbalance"
        if imbalance_ratio <= 3.0:
            return "moderate imbalance"
        if imbalance_ratio <= 6.0:
            return "strong imbalance"
        return "very strong imbalance"

    # function to format the size in bytes to a human readable format
    @staticmethod
    def format_bytes(size_in_bytes: int) -> str:
        if size_in_bytes < 1024:
            return f"{size_in_bytes} B"
        if size_in_bytes < 1024**2:
            return f"{size_in_bytes / 1024:.2f} KB"
        return f"{size_in_bytes / (1024**2):.2f} MB"

    # function to create a markdown anchor from a text
    @staticmethod
    def markdown_anchor(text: str) -> str:
        anchor = text.lower().strip()
        anchor = re.sub(r"[^\w\s-]", "", anchor)
        anchor = re.sub(r"\s+", "-", anchor)
        return anchor

    # function to get the min, max, and mean of a series
    @staticmethod
    def series_min_max_mean(series: pd.Series) -> str:
        if series.empty:
            return "N/A"
        return f"{float(series.min()):.4f} / {float(series.max()):.4f} / {float(series.mean()):.4f}"

    # function to format a timedelta to a human readable format
    @staticmethod
    def format_timedelta(value: timedelta | pd.Timedelta) -> str:
        if pd.isna(value):
            return "N/A"
        total_seconds = int(value.total_seconds())
        days, remainder = divmod(total_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        parts = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")
        if seconds or not parts:
            parts.append(f"{seconds}s")
        return " ".join(parts)

    # function to calculate the average word count of a series
    @staticmethod
    def average_word_count(series: pd.Series) -> float:
        if series.empty:
            return 0.0
        text = series.astype(str).str.strip()
        words = text.str.split().str.len()
        return float(words.mean()) if len(words) else 0.0

    # function to normalize a boolean value to 1 and 0
    @staticmethod
    def normalize_bool_token(value: Any) -> int | None:
        if pd.isna(value):
            return None
        normalized = str(value).strip().lower()
        truthy = {"1", "true", "t", "yes", "y"}
        falsy = {"0", "false", "f", "no", "n"}
        if normalized in truthy:
            return 1
        if normalized in falsy:
            return 0
        return None

    # function to check if a series is a boolean series
    def is_boolean_series(self, series: pd.Series) -> bool:
        if series.empty:
            return False
        if pd.api.types.is_bool_dtype(series):
            return True
        normalized = series.map(self.normalize_bool_token).dropna()
        if len(normalized) != len(series):
            return False
        return normalized.nunique(dropna=True) == 2

    # function to calculate the datetime parse success ratio
    # checks if a string column is a datetime in reality
    def datetime_parse_success_ratio(self, series: pd.Series) -> float:
        if series.empty:
            return 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            try:
                parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            except TypeError:
                try:
                    parsed = pd.to_datetime(series, errors="coerce")
                except Exception:
                    return 0.0
            except Exception:
                return 0.0
        success = parsed.notna().sum()
        return float(success / len(series)) if len(series) else 0.0

    # function to try and convert a string column to a numeric column
    # checks if a string column is a numeric column in reality
    def try_convert_string_numeric(self, col: str) -> bool:
        if self.df is None:
            return False
        series = self.df[col]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            return False

        non_null = series.dropna().astype(str)
        if non_null.empty:
            return False

        cleaned = (
            non_null.str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.strip()
        )
        converted = pd.to_numeric(cleaned, errors="coerce")
        success_ratio = converted.notna().mean() if len(converted) else 0.0
        if success_ratio > 0.90:
            self.add_warning(
                f"Column '{col}' appears numeric-like in string format (parse success ratio: {success_ratio * 100:.2f}%)."
            )
            return True
        return False

    # function to calculate the correlation strength
    @staticmethod
    def corr_strength(value: float, categorical: bool = False) -> str:
        if value < 0.20:
            return "Very Weak"
        if value < 0.40:
            return "Weak"
        if value < 0.60:
            return "Moderate"
        if value < 0.80:
            return "Strong"
        return "Very Strong"

    # function to calculate the Cramer's V
    @staticmethod
    def cramers_v(x: pd.Series, y: pd.Series) -> float:
        data = pd.DataFrame({"x": x, "y": y}).dropna()
        if data.empty:
            return 0.0
        contingency = pd.crosstab(data["x"], data["y"])
        if contingency.empty:
            return 0.0
        chi2 = chi2_contingency(contingency)[0]
        n = contingency.values.sum()
        if n == 0:
            return 0.0
        r, k = contingency.shape
        phi2 = chi2 / n
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1)) if n > 1 else 0
        rcorr = r - ((r - 1) ** 2) / (n - 1) if n > 1 else r
        kcorr = k - ((k - 1) ** 2) / (n - 1) if n > 1 else k
        denom = min((kcorr - 1), (rcorr - 1))
        if denom <= 0:
            return 0.0
        return float(np.sqrt(phi2corr / denom))

    # function to write the report to a file
    def write_report(self) -> None:
        report = "\n\n".join(self.report_sections)
        with open(self.output_path, "w", encoding="utf-8") as file:
            file.write(report)