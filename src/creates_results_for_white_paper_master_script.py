# from step_00_load_and_clean_input import unzip_and_clean_data
# from step_01_feature_engineering import feature_engineer_data

# from step_02_visualization import (
#     visuals_for_report_hist_and_first_kde,
#     visuals_for_report_second_kde_and_data_dict,
#     generate_ydata_eda,
# )
# from step_03_linear_regression_approach import fit_model_one, fit_model_two
from step_04_lightgbm_approach_with_text_and_hyperopt import fit_model

# unzip_and_clean_data()
# feature_engineer_data()
# visuals_for_report_hist_and_first_kde()
# visuals_for_report_second_kde_and_data_dict()
# generate_ydata_eda("clean")
# generate_ydata_eda("raw")
# fit_model_one()
# fit_model_two()

# fit_model(model_type="basic", hyper_parm_tune=False)
fit_model(model_type="hyperopt", hyper_parm_tune=False)
