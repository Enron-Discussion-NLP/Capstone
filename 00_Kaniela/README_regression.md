# zillow_regression_project

This repository holds all of my work and deliverables for the Regression project.

---

# About the Project

## Project Goal
The goal of this project is to identify drivers of assessed home value for single family properties so that we can become a stronger competitor in the home-buying market.

## Project Description
Inaccurate predictions of home values have the potential to ruin us. The efficacy of our model could determine whether or not we can stay in the instant home-buying business. With this project, I will determine which attributes of single-family residential properties contribute most to their tax assessed value. Using our data from 2017, my goal is to develop a model that will predict these property values with more accuracy than our current model.

## Initial Questions
 1. Does a home's location affect how much it costs?
 2. Is there a relationship between number of bedrooms and number of bathrooms?
 3. Do bigger homes have more bathrooms?
 4. What drives tax_value? (Which features are most strongly correlated with tax value?)

## Data Dictionary
| Variable | Meaning |
| -------- | ------- |
| zillow   | Full, original dataframe retrieved from the zillow mySQL database |
| train    | Sample (56%) of zillow used for exploring data and fitting/training models|
| validate | Sample (24%) of zillow used to evaluate multiple models |
| test     | Sample (20%) of zillow used to evaluate the best model |
| tax_value | Our target variable: the property's tax assessed value |
| beds     | Number of bedrooms |
| baths    | Number of bathrooms, including fractional bathrooms |
| sq_ft    | Calculated total finished living area |
| fips     | Federal Information Processing Standard code |
| beds_scaled | Standard-scaled `beds` |
| baths_scaled | Standard-scaled `baths` |
| sq_ft_scaled | Standard-scaled `sq_ft` |
| x_train  | `train`, but only with scaled columns |
| y_train  | `train`, but only the target |
| x_validate | `validate`, but only with scaled columns |
| y_validate | `validate`, but only the target |
| x_test   | `test`, but only with scaled columns |
| y_test   | `test`, but only the target |


## The Plan

### Wrangle

1. Define a function to acquire the necessary zillow data from the mySQL database.
2. Define a function to clean the acquired zillow data.
3. Define a function to split the cleaned zillow data.
4. Define a function to scale the split zillow data.
5. Define a function to combine the previous steps into a single function.
6. Ensure all functions work properly and add them to wrangle.py file.

### Explore
1. Ask a clear question.
2. Develop null and alternative hypotheses for that question.
3. Use visualizations and statistical tests to find answers.
4. Clearly state the answer to the question and summarize findings.
5. Repeat for a total of at least 4 questions.
6. Summarize key findings, takeaways, and actions.

### Model
1. Select a metric to use for evaluating models and explain why that metric was chosen.
2. Create and evaluate a baseline model.
    - Find mean value of target
    - Set all predictions to that value
    - Evaluate based on selected evaluation metric
3. Develop three models.
4. Evaluate all three models on the train sample, note observations.
5. Evaluate the top two models on the validate sample, note observations.
6. Evaluate the top performing model on the test sample, note observations.

### Deliver
1. Ensure final report notebook is thoroughly code commented.
2. Ensure notebook contains enough Markdown cells to guide the reader through the report.
3. Write a conclusion summary.
4. Develop actionable recommendations.
5. Suggest next steps for research and/or model improvement.
6. Run final report notebook from beginning to be sure that there are no errors.
7. Submit link to repository containing project files.
8. Deliver live presentation.

## Steps to Reproduce
- In a local repository, create a .gitignore file and add env.py.
- In that same repository, create an env.py file to store the hostname, username, and password that will be used to access the zillow mySQL database.
- Clone this repository, making sure the .gitignore is successfully hiding the env.py file.
- Run all cells in Report.ipynb.
- If desired, run all cells in the other Jupyter notebooks (wrangle.ipynb, explore.ipynb) in this repository to see my full, in-depth step-by-step process for this project.

---
## Conclusion

### Summary
There is absolutely room for improvement when it comes to our current model for predicting home values. By exploring the data we currently have, I was able to create a model with an RMSE value of 217301. While this is helpful, I know it can still be improved.

### Recommendation
Find a way to quantify subjective attributes of the home.
 - Nearby Attractions
 - Neighbor Quality
 - Neighborhood Safety

### Next Steps
With more time, I would like to dive deeper into the zillow database and implement feature engineering to discover the best combination of available features to predict home values. There is a ton of data in there and I'm sure we can use it to our advantage.
