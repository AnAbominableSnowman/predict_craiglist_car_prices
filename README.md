# video_game_sales_predictions

This is a repo I created as product of work for various Data jobs. In this repo, I take 450k Craigslist ads and try to predict each vehicles ad price. Linear regression failed in this project but I had a lot of success with Light GBM using HyperOpt and TF_IDF to encode info from the descriptions. Getting an RMSE of \$5,400 and successfully explaining about 85% of the variation in price. For a walk through of the process and results for a semi-technical to technical audience, see the [white paper](White Paper.md). 

### Understanding the code
We have 5 steps here. Clean data, feature engineer data, visualize data, fit a linear regression model, fit a light GBM model and finally, use SHAP to analyze the model features. I have two master scripts that run the data. One is [creates_results_for_white_paper_master_script](creates_results_for_white_paper_master_script) which is a higher level paper that dupilcates the results you will see in the white paper. For those for feeling more adventourous and interested in diving into the code, [master_script](master_script.py) removes a layer of abstraction. 


### Decisions points in my analysis.
I wish I'd have been better at saving off the results that made me pursue certain paths or stopped me from further pursuing other paths. Some results are still likely in the past commits but if you have questions, feel free to leave and issue. 

### FAQ on running it:
Requirements are stored in requirements.txt and I'm using Python 3.11.9 here. Y_data's profiling report is the best at easy but incredible EDA visualizing but it's a finicky beast. If you are running into issues, not running `generate_ydata_eda` in `src/step_02_visualization.py` will drastically reduce your number of installations and issues. 

Where are all your .parquet files? I have them ignored for git purposes. They are particularly large and hit GH's 100mb limit. The proccessing steps are very fast, less then a minute on my rather average laptop, so the juice isn't worth the squeeze.

I have throughly commented most functions and the white paper also covers a lot so this readme will be sparse. As always, if you have further questions, feel free to post them as issues. 

What style is this code? I use astral's Ruff. I can't recommend it enough. 

Any VENV? Yes, I have 