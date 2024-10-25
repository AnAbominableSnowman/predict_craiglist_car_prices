# from step_02_visualization import (
#     #     visuals_for_report_hist_and_first_kde,
#     #     visuals_for_report_second_kde_and_data_dict,
#     generate_ydata_eda,
# )
# from step_03_linear_regression_approach import fit_model_one, fit_model_two
# from step_04_lightgbm_approach_with_text_and_hyperopt import (
#     # fit_model_three,
#     fit_model_four,
# )
from step_05_shap_analysis import plot_shap_summary


# unzip_and_clean_data()
# feature_engineer_data()
# visuals_for_report_hist_and_first_kde()
# visuals_for_report_second_kde_and_data_dict()
# generate_ydata_eda("clean")
# generate_ydata_eda("raw")
# fit_model_one()
# fit_model_two()

# fit_model_three()
# fit_model_four(hyper_parm_tune=True)

plot_shap_summary(
    model_path="results/light_gbm__hyperopt_and_feature_engineering/best_lightgbm_model.pkl",
    data_path="intermediate_data/cleaned_edited_feature_engineered_input.parquet",
    output_dir="results/light_gbm__hyperopt_and_feature_engineering/",
    col_subset=None,
)
