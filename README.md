# mcr-ml-var v0.1.0

## Updates

**New Features**
- VAR model ensemble wrapper with XGBoost integration, 40% better forecasting accuracy on multivariate datasets
- Automated lag selection using AIC/BIC criteria, eliminates manual hyperparameter tuning
- Cross-validation splits for time series data, prevents temporal data leakage issues
- Parallel model fitting across CPU cores, scales with available hardware

**QoL**  
- Vectorized preprocessing pipeline, 25% faster feature engineering
- Enhanced error metrics with MSE/MAE/RMSE reporting, better model evaluation
- Proper logging implementation, replaced print debugging statements
- Type hints added throughout codebase, improved developer experience

**Service**
- Memory optimization for large datasets, handles 10x larger time series
- Stationarity test reliability improvements, works on edge case datasets  
- Missing value handling in preprocessing, no more pipeline crashes
- Enhanced exception messages, easier debugging for end users

**Deployment**
- Pinned dependency versions in requirements.txt, reproducible environment setup
- CI/CD pipeline stability fixes, eliminated random build failures
- Docker container support, consistent deployment across environments  
- Updated documentation with code examples, reduced onboarding time

**Bugfixes**
- MCR-ALS non-negativity constraints NaN generation, matrix decomposition now stable
- Memory leak in multivariate forecasting loop, long-running processes fixed
- Single-column DataFrame crash in model.fit(), handles edge case inputs
- Grid search infinite loops in hyperparameter optimization, timeout mechanisms added


**Changelog** :https://github.com/abhilashongit/mcr-ml-var/commits/NewModel**
