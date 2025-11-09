"""
Setup script for Customer Segmentation System.
This script creates the initial directory structure and files.
"""

import os
import sys


def create_project_structure():
    """
    Create the initial directory structure and files for the project.
    """
    # Create project directory structure
    directories = [
        "data/raw",
        "data/processed",
        "docs",
        "notebooks",
        "src",
        "models",
        "reports/figures",
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

    # Create the directory structure for src
    src_directories = [
        "src/data",
        "src/features",
        "src/models",
        "src/visualization",
        "src/utils",
    ]

    for directory in src_directories:
        os.makedirs(directory, exist_ok=True)
        # Create __init__.py files
        with open(f"{directory}/__init__.py", "w") as f:
            f.write(f"# {directory} module\n")
        print(f"Created {directory}")

    # Create main package __init__.py
    with open("src/__init__.py", "w") as f:
        f.write("# Customer Segmentation package\n")

    # Create empty files
    files = ["requirements.txt", "README.md", ".gitignore", "run.py"]
    for file in files:
        with open(file, "w") as f:
            pass
        print(f"Created file: {file}")

    # Write requirements
    with open("requirements.txt", "w") as f:
        f.write(
            """
            numpy==1.24.3
            pandas==2.0.2
            matplotlib==3.7.1
            seaborn==0.12.2
            plotly==5.15.0
            scikit-learn==1.2.2
            streamlit==1.24.0
            jupyter==1.0.0
            """
        )

    # Create basic README
    with open("README.md", "w") as f:
        f.write(
            """# Customer Segmentation System

        A modular, extensible customer segmentation system that analyzes customer purchasing behavior and creates actionable customer segments. Built using Python, Streamlit, and scikit-learn.

        ## Features

        - **Data Loading & Preprocessing**: Clean and preprocess raw retail transaction data
        - **Feature Engineering**: Create meaningful customer-level features
        - **Customer Clustering**: Segment customers using KMeans clustering
        - **Segment Interpretation**: Automatically interpret clusters and create customer personas
        - **Dynamic Segmentation**: Update segments as new data becomes available
        - **Interactive Visualization**: Explore segments through an intuitive Streamlit dashboard
        - **Marketing Recommendations**: Get actionable marketing strategies for each segment

        ## Project Structure

        ```
        .
        ├── data/                 # Data directory
        │   ├── raw/              # Raw data files
        │   └── processed/        # Processed data files
        ├── models/               # Saved models
        ├── reports/              # Reports and visualizations
        │   └── figures/          # Generated figures
        ├── src/                  # Source code
        │   ├── data/             # Data loading and preprocessing
        │   ├── features/         # Feature engineering
        │   ├── models/           # Clustering models
        │   ├── visualization/    # Visualization and dashboard
        │   └── utils/            # Helper utilities
        ├── requirements.txt      # Project dependencies
        ├── README.md             # Project documentation
        └── run.py                # Script to run the application
        ```
        """
        )

    # Create a basic .gitignore
    with open(".gitignore", "w") as f:
        f.write(
            """# Python
        __pycache__/
        *.py[cod]
        *$py.class
        *.so
        .Python
        env/
        build/
        develop-eggs/
        dist/
        downloads/
        eggs/
        .eggs/
        lib/
        lib64/
        parts/
        sdist/
        var/
        *.egg-info/
        .installed.cfg
        *.egg

        # Jupyter Notebook
        .ipynb_checkpoints

        # Data files - uncomment if you want to ignore data files
        # data/raw/
        # data/processed/

        # Model files - uncomment if you want to ignore model files
        # models/

        # Virtual Environment
        venv/
        env/
        ENV/

        # IDE files
        .idea/
        .vscode/
        *.swp
        *.swo
        """
        )

    # Create MIT license file
    with open("LICENSE.txt", "w") as f:
        f.write(
            """MIT License
            Copyright (c) 2025 Nguyen Thai Ha

            Permission is hereby granted, free of charge, to any person obtaining a copy
            of this software and associated documentation files (the "Software"), to deal
            in the Software without restriction, including without limitation the rights
            to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
            copies of the Software, and to permit persons to whom the Software is
            furnished to do so, subject to the following conditions:

            The above copyright notice and this permission notice shall be included in
            all copies or substantial portions of the Software.

            THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
            IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
            FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
            AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
            LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
            OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
            THE SOFTWARE.
            """
        )


if __name__ == "__main__":
    create_project_structure()
