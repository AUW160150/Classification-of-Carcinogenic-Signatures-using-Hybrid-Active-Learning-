# Classification-of-Carcinogenic-Signatures-using-Hybrid-Active-Learning-
Active Learning Framework for Tabular Classification that demonstrates how to iteratively improve machine learning models by intelligently selecting the most informative samples for labeling. Instead of randomly sampling data, it uses three sophisticated strategies to find samples that will most improve model performance.
Individual Sections and Their Purpose:
1. Configuration & Setup (Lines 1-120)

Purpose: Centralizes all hyperparameters and utility functions
Key Components:

RunConfig dataclass for experiment configuration
set_seed() for reproducible results
setup_logging() for debugging and monitoring


Demonstrates: Clean configuration management and reproducibility best practices

2. Data Loading & Preprocessing (Lines 121-280)

Purpose: Robust data handling with comprehensive error checking
Key Functions:

load_csv(): Safe file loading with validation
split_features(): Automatic feature type inference
build_preprocessor(): ML preprocessing pipeline creation


Demonstrates: Exception handling, data validation, and modular preprocessing

3. Model Building & Cross-Validation (Lines 281-380)

Purpose: Model creation and performance evaluation utilities
Key Functions:

build_model(): Creates scikit-learn pipelines
cv_metrics(): Stratified cross-validation evaluation


Demonstrates: Pipeline construction and proper model evaluation techniques

4. Active Learning Strategies (Lines 381-580)

Purpose: Three complementary sampling strategies for active learning
Key Components:

uncertainty_margin(): Selects samples where model is least confident
vote_entropy(): Uses committee disagreement to find informative samples
diversity_kmeans(): Ensures coverage of different feature space regions


Demonstrates: Advanced sampling algorithms and ensemble methods

5. Active Learning Runner (Lines 581-780)

Purpose: Main orchestrator that coordinates the entire learning process
Key Features:

Manages labeled/unlabeled data splits
Implements hybrid sampling strategy
Tracks performance over iterations


Demonstrates: Complex workflow orchestration and state management

6. CLI & Exception Handling (Lines 781-950)

Purpose: Command-line interface with comprehensive error handling
Key Features:

Robust exception handling for various error types
User-friendly error messages and recovery suggestions
Comprehensive argument parsing


Demonstrates: Production-ready error handling and user experience design

7. Unit Tests (Lines 951-1100)

Purpose: Validates core functionality with automated tests
Test Coverage:

test_uncertainty_margin(): Tests uncertainty sampling logic
test_vote_entropy(): Validates committee voting calculations
test_diversity_kmeans(): Checks clustering-based selection


Demonstrates: Effective unit testing strategies for ML components

Key Improvements Made:

PEP8 Compliance:

Proper naming conventions (snake_case for functions/variables)
Consistent indentation and spacing
Line length management
Import organization


Enhanced Documentation:

Comprehensive docstrings with Args/Returns/Examples
Type hints throughout
Inline comments explaining complex logic
Clear section headers


Better Exception Handling:

Specific exception types for different error categories
Informative error messages with recovery suggestions
Graceful degradation and early stopping


Improved Modularity:

Clear separation of concerns
Reusable components
Configuration externalization
Pluggable sampling strategies


Debugging Features:

Comprehensive logging at appropriate levels
Progress tracking and iteration reporting
Performance history tracking



Unit Tests Section (Lines 951-1050):
The unit tests demonstrate testing best practices for ML components:

Controlled test data with known expected outcomes
Edge case handling (empty data, boundary conditions)
Assertion-based validation with clear error messages
Isolated testing of individual components

Exception Handling Examples (Lines 781-880):
The cmd_run() function showcases comprehensive error handling:

Data errors: Missing files, invalid columns, empty datasets
Configuration errors: Invalid parameters, incompatible settings
Model errors: Fitting failures, memory issues
Recovery guidance: Specific suggestions for each error type
