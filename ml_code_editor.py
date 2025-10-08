import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import ast
import sys
import io
import contextlib
import threading
import time
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import pandas as pd
from sklearn.datasets import make_classification, make_regression, load_iris, fetch_california_housing
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

class CodeBlock:
    """Represents a code block that was executed"""
    def __init__(self, canvas, code_type, line_number, x, y, width=150, height=50):
        self.canvas = canvas
        self.code_type = code_type
        self.line_number = line_number
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.is_executing = False
        
        # Colors for different code types
        self.colors = {
            'import': '#607D8B',        # Blue Grey
            'data_load': '#4CAF50',     # Green
            'preprocessing': '#2196F3', # Blue
            'model': '#FF9800',         # Orange
            'training': '#E91E63',      # Pink
            'prediction': '#9C27B0',    # Purple
            'visualization': '#F44336', # Red
            'evaluation': '#795548',    # Brown
            'other': '#757575'          # Grey
        }
        
        self.create_block()
    
    def create_block(self):
        """Create visual block on canvas"""
        color = self.colors.get(self.code_type, '#757575')
        
        # Main rectangle
        self.rect_id = self.canvas.create_rectangle(
            self.x, self.y, self.x + self.width, self.y + self.height,
            fill=color, outline='white', width=2, tags=f"block_{id(self)}"
        )
        
        # Block text
        display_text = f"Line {self.line_number}\n{self.code_type.replace('_', ' ').title()}"
        self.text_id = self.canvas.create_text(
            self.x + self.width//2, self.y + self.height//2,
            text=display_text, fill='white', font=('Arial', 9, 'bold'),
            tags=f"block_{id(self)}"
        )
        
        # Connection points
        self.create_connection_points()
    
    def create_connection_points(self):
        """Create input/output connection points"""
        point_size = 6
        
        # Input point (top)
        self.input_point = self.canvas.create_oval(
            self.x + self.width//2 - point_size//2, self.y - point_size//2,
            self.x + self.width//2 + point_size//2, self.y + point_size//2,
            fill='lightblue', outline='blue', width=2,
            tags=f"block_{id(self)}"
        )
        
        # Output point (bottom)
        self.output_point = self.canvas.create_oval(
            self.x + self.width//2 - point_size//2, self.y + self.height - point_size//2,
            self.x + self.width//2 + point_size//2, self.y + self.height + point_size//2,
            fill='lightgreen', outline='green', width=2,
            tags=f"block_{id(self)}"
        )
    
    def highlight(self, color='#FFD700'):
        """Highlight block during execution"""
        self.canvas.itemconfig(self.rect_id, outline=color, width=4)
        self.is_executing = True
    
    def unhighlight(self):
        """Remove highlight"""
        self.canvas.itemconfig(self.rect_id, outline='white', width=2)
        self.is_executing = False

class MLCodeEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("🐍 ML Code Editor - Live Visualization")
        self.root.geometry("1800x1000")
        self.root.configure(bg='#1e1e1e')
        
        # State variables
        self.code_blocks = []
        self.execution_globals = {}
        self.current_line = 0
        self.is_running = False
        self.dev_focus = False
        self.block_tag_map = {}
        self.tooltip_window = None
        self.canvas_scale = 1.0
        self.is_panning = False
        self.pan_start = (0, 0)

        # Layers animation state (Conv→ReLU→Pool)
        self.layers_grid_size = 28
        self.layers_img = np.zeros((28, 28), dtype=float)
        self.layers_anim_running = False
        self.layers_anim_speed_ms = 120
        self.layers_speed_var = tk.IntVar(value=120)
        self.layers_conv_kernel = np.array([[1, 0, -1],
                                            [1, 0, -1],
                                            [1, 0, -1]], dtype=float)
        self.layers_conv_map = None
        self.layers_relu_map = None
        self.layers_pool_map = None
        self.layers_scan_index = 0
        self.layers_scan_positions = []
        self.layers_input_rect = None
        self.layers_axes = {}
        self.layers_fig = None
        self.layers_plot_canvas = None
        self.layers_pad_canvas = None
        self.layers_tab = None
        self.layers_flat = None
        self.layers_dense_W = None
        self.layers_dense_b = None
        self.layers_logits = None
        self.layers_probs = None
        self.layers_dense_step = 0
        
        # Initialize execution environment
        self.setup_execution_environment()
        self.setup_ui()
        self.load_sample_code()
        self.setup_keybindings()
    
    def setup_execution_environment(self):
        """Setup the execution environment with ML libraries"""
        self.execution_globals = {
            'np': np,
            'pd': pd,
            'plt': plt,
            'make_classification': make_classification,
            'make_regression': make_regression,
            'load_iris': load_iris,
            'fetch_california_housing': fetch_california_housing,
            'LinearRegression': LinearRegression,
            'LogisticRegression': LogisticRegression,
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'DecisionTreeRegressor': DecisionTreeRegressor,
            'RandomForestClassifier': RandomForestClassifier,
            'RandomForestRegressor': RandomForestRegressor,
            'train_test_split': train_test_split,
            'accuracy_score': accuracy_score,
            'mean_squared_error': mean_squared_error,
            'classification_report': classification_report,
            'StandardScaler': StandardScaler,
            'LabelEncoder': LabelEncoder,
            'print': self.custom_print,
            '__builtins__': __builtins__
        }
    
    def setup_ui(self):
        """Setup the main UI"""
        # Main container
        main_frame = tk.Frame(self.root, bg='#1e1e1e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.main_frame = main_frame
        
        # Top toolbar
        self.create_toolbar(main_frame)
        
        # Main content area
        content_frame = tk.Frame(main_frame, bg='#1e1e1e')
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.content_frame = content_frame
        
        # Create main panels
        self.create_library_explorer(content_frame)
        self.create_code_editor(content_frame)
        self.create_block_canvas(content_frame)
        self.create_output_panel(content_frame)
        
        # Bottom panel - Console and status
        self.create_console_panel(main_frame)
    
    def create_toolbar(self, parent):
        """Create the top toolbar"""
        toolbar = tk.Frame(parent, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        toolbar.pack(fill=tk.X, pady=(0, 5))
        self.toolbar = toolbar
        
        # Control buttons
        tk.Button(toolbar, text="▶ Run Code", command=self.run_code,
                 bg='#4CAF50', fg='white', font=('Arial', 12, 'bold')).pack(side=tk.LEFT, padx=5, pady=5)
        
        tk.Button(toolbar, text="⏸ Step by Step", command=self.run_step_by_step,
                 bg='#2196F3', fg='white', font=('Arial', 12, 'bold')).pack(side=tk.LEFT, padx=5, pady=5)
        
        tk.Button(toolbar, text="⏹ Stop", command=self.stop_execution,
                 bg='#F44336', fg='white', font=('Arial', 12, 'bold')).pack(side=tk.LEFT, padx=5, pady=5)
        
        tk.Button(toolbar, text="🔄 Reset", command=self.reset_environment,
                 bg='#FF9800', fg='white', font=('Arial', 12, 'bold')).pack(side=tk.LEFT, padx=5, pady=5)
        
        tk.Button(toolbar, text="💾 Save", command=self.save_code,
                 bg='#9C27B0', fg='white', font=('Arial', 12, 'bold')).pack(side=tk.LEFT, padx=5, pady=5)
        
        tk.Button(toolbar, text="📁 Load", command=self.load_code,
                 bg='#607D8B', fg='white', font=('Arial', 12, 'bold')).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Add Templates button to toolbar
        templates_btn = tk.Button(toolbar, text="📋 Templates", 
                                command=self.show_templates,
                                bg='#4CAF50', fg='white', 
                                font=('Consolas', 10),
                                relief=tk.FLAT, padx=10)
        templates_btn.pack(side=tk.LEFT, padx=5)

        # Dev Focus toggle
        self.dev_focus_btn = tk.Button(toolbar, text="🎯 Dev Focus: OFF",
                                       command=self.toggle_dev_focus,
                                       bg='#3E3E3E', fg='white',
                                       font=('Consolas', 10), relief=tk.FLAT, padx=10)
        self.dev_focus_btn.pack(side=tk.LEFT, padx=5)

        # Layers animation quick access
        tk.Button(toolbar, text="🧠 Layers", command=self.open_layers_tab_and_run,
                 bg='#795548', fg='white', font=('Arial', 12, 'bold')).pack(side=tk.LEFT, padx=5, pady=5)
        
        # 3D visualizer has been temporarily disabled per request

        # Removed Model Builder toggle per request; 3D view focuses on visualization only
        
        # Speed control
        tk.Label(toolbar, text="Execution Speed:", bg='#2d2d2d', fg='white', 
                font=('Arial', 10)).pack(side=tk.RIGHT, padx=(20, 5), pady=5)
        
        self.speed_var = tk.DoubleVar(value=1.0)
        self.speed_scale = tk.Scale(toolbar, from_=0.1, to=3.0, resolution=0.1, 
                                   orient=tk.HORIZONTAL, variable=self.speed_var,
                                   bg='#2d2d2d', fg='white', font=('Arial', 9))
        self.speed_scale.pack(side=tk.RIGHT, padx=5, pady=5)
    
    def create_library_explorer(self, parent):
        """Tạo Library Explorer panel cho người mới học ML"""
        # Library Explorer Frame
        lib_frame = tk.Frame(parent, bg='#2b2b2b', width=250)
        lib_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        lib_frame.pack_propagate(False)
        self.lib_frame = lib_frame
        
        # Title
        title_label = tk.Label(lib_frame, text="📚 ML Libraries", 
                              bg='#2b2b2b', fg='#ffffff', 
                              font=('Consolas', 12, 'bold'))
        title_label.pack(pady=(10, 5))
        
        # Scrollable frame for libraries
        canvas = tk.Canvas(lib_frame, bg='#2b2b2b', highlightthickness=0)
        scrollbar = tk.Scrollbar(lib_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#2b2b2b')

        # Keep scrollregion updated and inner frame width aligned with canvas width
        def _on_frame_configure(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
        scrollable_frame.bind("<Configure>", _on_frame_configure)

        window_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        def _on_canvas_configure(event):
            # Ensure the inner frame width always matches the canvas width
            canvas.itemconfigure(window_id, width=event.width)
        canvas.bind('<Configure>', _on_canvas_configure)

        canvas.configure(yscrollcommand=scrollbar.set)

        # Mouse wheel scrolling support (Windows/Mac/Linux)
        def _on_mousewheel(event):
            # Windows/Mac usually provide event.delta as multiples of 120
            try:
                delta = int(-1 * (event.delta / 120))
            except Exception:
                delta = -1 if getattr(event, 'delta', 1) > 0 else 1
            canvas.yview_scroll(delta, 'units')

        def _on_linux_scroll_up(event):
            canvas.yview_scroll(-1, 'units')

        def _on_linux_scroll_down(event):
            canvas.yview_scroll(1, 'units')

        # Bind mouse wheel events only when cursor is inside the scrollable area
        scrollable_frame.bind('<Enter>', lambda e: canvas.bind_all('<MouseWheel>', _on_mousewheel))
        scrollable_frame.bind('<Leave>', lambda e: canvas.unbind_all('<MouseWheel>'))
        scrollable_frame.bind('<Button-4>', _on_linux_scroll_up)
        scrollable_frame.bind('<Button-5>', _on_linux_scroll_down)
        
        # ML Libraries data với detailed methods - COMPLETE VERSION
        libraries = {
            "📊 Data Manipulation": {
                "pandas": {
                    "desc": "Data analysis and manipulation",
                    "example": "import pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df.head())",
                    "methods": {
                        "Data Loading": {
                            "pd.read_csv()": "df = pd.read_csv('file.csv')",
                            "pd.read_excel()": "df = pd.read_excel('file.xlsx')",
                            "pd.read_json()": "df = pd.read_json('file.json')",
                            "pd.read_sql()": "df = pd.read_sql('SELECT * FROM table', connection)",
                            "pd.DataFrame()": "df = pd.DataFrame({'col1': [1,2,3], 'col2': [4,5,6]})"
                        },
                        "Data Exploration": {
                            "df.head()": "df.head(10)  # First 10 rows",
                            "df.tail()": "df.tail(5)   # Last 5 rows", 
                            "df.info()": "df.info()    # Dataset info",
                            "df.describe()": "df.describe()  # Statistical summary",
                            "df.shape": "print(df.shape)  # (rows, columns)",
                            "df.columns": "print(df.columns)  # Column names",
                            "df.dtypes": "print(df.dtypes)  # Data types",
                            "df.nunique()": "df.nunique()  # Unique values count"
                        },
                        "Data Cleaning": {
                            "df.isnull()": "df.isnull().sum()  # Count missing values",
                            "df.dropna()": "df.dropna()  # Remove missing values",
                            "df.fillna()": "df.fillna(0)  # Fill missing with 0",
                            "df.drop_duplicates()": "df.drop_duplicates()  # Remove duplicates",
                            "df.replace()": "df.replace({'old': 'new'})  # Replace values"
                        },
                        "Data Selection": {
                            "df['column']": "df['column_name']  # Select column",
                            "df.loc[]": "df.loc[0:5, 'col1':'col3']  # Label-based selection",
                            "df.iloc[]": "df.iloc[0:5, 0:3]  # Position-based selection",
                            "df.query()": "df.query('age > 25')  # Query with condition",
                            "df.filter()": "df.filter(regex='.*_score')  # Filter columns"
                        },
                        "Data Transformation": {
                            "df.groupby()": "df.groupby('category').mean()  # Group and aggregate",
                            "df.pivot_table()": "df.pivot_table(values='sales', index='date', columns='product')",
                            "df.merge()": "pd.merge(df1, df2, on='key')  # Join dataframes",
                            "df.concat()": "pd.concat([df1, df2])  # Concatenate dataframes",
                            "df.apply()": "df.apply(lambda x: x*2)  # Apply function"
                        }
                    }
                },
                "numpy": {
                    "desc": "Numerical computing with arrays",
                    "example": "import numpy as np\narr = np.array([1, 2, 3])\nprint(np.mean(arr))",
                    "methods": {
                        "Array Creation": {
                            "np.array()": "np.array([1, 2, 3, 4])",
                            "np.zeros()": "np.zeros((3, 4))  # 3x4 array of zeros",
                            "np.ones()": "np.ones((2, 3))   # 2x3 array of ones",
                            "np.arange()": "np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]",
                            "np.linspace()": "np.linspace(0, 1, 5)  # 5 points from 0 to 1",
                            "np.eye()": "np.eye(3)  # 3x3 identity matrix",
                            "np.random.rand()": "np.random.rand(3, 3)  # Random 3x3 array"
                        },
                        "Array Operations": {
                            "np.mean()": "np.mean(arr)  # Average",
                            "np.std()": "np.std(arr)   # Standard deviation", 
                            "np.sum()": "np.sum(arr)   # Sum of elements",
                            "np.max()": "np.max(arr)   # Maximum value",
                            "np.min()": "np.min(arr)   # Minimum value",
                            "np.median()": "np.median(arr)  # Median value",
                            "np.var()": "np.var(arr)   # Variance"
                        },
                        "Array Manipulation": {
                            "np.reshape()": "arr.reshape(2, 3)  # Change shape",
                            "np.transpose()": "np.transpose(arr)  # Transpose",
                            "np.concatenate()": "np.concatenate([arr1, arr2])",
                            "np.split()": "np.split(arr, 3)  # Split into 3 parts",
                            "np.flatten()": "arr.flatten()  # Flatten to 1D",
                            "np.sort()": "np.sort(arr)  # Sort array"
                        },
                        "Linear Algebra": {
                            "np.dot()": "np.dot(a, b)  # Matrix multiplication",
                            "np.linalg.inv()": "np.linalg.inv(matrix)  # Matrix inverse",
                            "np.linalg.det()": "np.linalg.det(matrix)  # Determinant",
                            "np.linalg.eig()": "np.linalg.eig(matrix)  # Eigenvalues"
                        }
                    }
                }
            },
            "🤖 Machine Learning": {
                "sklearn": {
                    "desc": "Machine learning algorithms and tools",
                    "example": "from sklearn.linear_model import LinearRegression\nmodel = LinearRegression().fit(X, y)",
                    "methods": {
                        "Classification Models": {
                            "LogisticRegression": "from sklearn.linear_model import LogisticRegression\nmodel = LogisticRegression()",
                            "DecisionTreeClassifier": "from sklearn.tree import DecisionTreeClassifier\nmodel = DecisionTreeClassifier()",
                            "RandomForestClassifier": "from sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier()",
                            "SVC": "from sklearn.svm import SVC\nmodel = SVC()",
                            "KNeighborsClassifier": "from sklearn.neighbors import KNeighborsClassifier\nmodel = KNeighborsClassifier()",
                            "GaussianNB": "from sklearn.naive_bayes import GaussianNB\nmodel = GaussianNB()"
                        },
                        "Regression Models": {
                            "LinearRegression": "from sklearn.linear_model import LinearRegression\nmodel = LinearRegression()",
                            "DecisionTreeRegressor": "from sklearn.tree import DecisionTreeRegressor\nmodel = DecisionTreeRegressor()",
                            "RandomForestRegressor": "from sklearn.ensemble import RandomForestRegressor\nmodel = RandomForestRegressor()",
                            "SVR": "from sklearn.svm import SVR\nmodel = SVR()",
                            "Ridge": "from sklearn.linear_model import Ridge\nmodel = Ridge()",
                            "Lasso": "from sklearn.linear_model import Lasso\nmodel = Lasso()"
                        },
                        "Clustering": {
                            "KMeans": "from sklearn.cluster import KMeans\nkmeans = KMeans(n_clusters=3).fit(X)",
                            "DBSCAN": "from sklearn.cluster import DBSCAN\ndbscan = DBSCAN().fit(X)",
                            "AgglomerativeClustering": "from sklearn.cluster import AgglomerativeClustering\nagg = AgglomerativeClustering()"
                        },
                        "Preprocessing": {
                            "StandardScaler": "from sklearn.preprocessing import StandardScaler\nscaler = StandardScaler().fit_transform(X)",
                            "MinMaxScaler": "from sklearn.preprocessing import MinMaxScaler\nscaler = MinMaxScaler().fit_transform(X)",
                            "LabelEncoder": "from sklearn.preprocessing import LabelEncoder\nle = LabelEncoder().fit_transform(y)",
                            "OneHotEncoder": "from sklearn.preprocessing import OneHotEncoder\nenc = OneHotEncoder().fit_transform(X)",
                            "PolynomialFeatures": "from sklearn.preprocessing import PolynomialFeatures\npoly = PolynomialFeatures(degree=2)"
                        },
                        "Model Selection": {
                            "train_test_split": "from sklearn.model_selection import train_test_split\nX_train, X_test, y_train, y_test = train_test_split(X, y)",
                            "cross_val_score": "from sklearn.model_selection import cross_val_score\nscores = cross_val_score(model, X, y, cv=5)",
                            "GridSearchCV": "from sklearn.model_selection import GridSearchCV\ngrid = GridSearchCV(model, param_grid)",
                            "RandomizedSearchCV": "from sklearn.model_selection import RandomizedSearchCV\nrandom_search = RandomizedSearchCV(model, param_distributions)"
                        },
                        "Metrics": {
                            "accuracy_score": "from sklearn.metrics import accuracy_score\naccuracy_score(y_true, y_pred)",
                            "classification_report": "from sklearn.metrics import classification_report\nprint(classification_report(y_true, y_pred))",
                            "confusion_matrix": "from sklearn.metrics import confusion_matrix\nconfusion_matrix(y_true, y_pred)",
                            "mean_squared_error": "from sklearn.metrics import mean_squared_error\nmse = mean_squared_error(y_true, y_pred)",
                            "r2_score": "from sklearn.metrics import r2_score\nr2 = r2_score(y_true, y_pred)",
                            "roc_auc_score": "from sklearn.metrics import roc_auc_score\nauc = roc_auc_score(y_true, y_pred)"
                        }
                    }
                },
                "xgboost": {
                    "desc": "Gradient boosting framework",
                    "example": "import xgboost as xgb\nmodel = xgb.XGBClassifier().fit(X_train, y_train)",
                    "methods": {
                        "Models": {
                            "XGBClassifier": "import xgboost as xgb\nmodel = xgb.XGBClassifier()",
                            "XGBRegressor": "import xgboost as xgb\nmodel = xgb.XGBRegressor()",
                            "XGBRanker": "import xgboost as xgb\nmodel = xgb.XGBRanker()"
                        },
                        "Training": {
                            "model.fit()": "model.fit(X_train, y_train)",
                            "model.predict()": "predictions = model.predict(X_test)",
                            "model.predict_proba()": "probabilities = model.predict_proba(X_test)"
                        },
                        "Feature Importance": {
                            "model.feature_importances_": "importances = model.feature_importances_",
                            "xgb.plot_importance()": "xgb.plot_importance(model)"
                        }
                    }
                },
                "tensorflow": {
                    "desc": "Deep learning framework",
                    "example": "import tensorflow as tf\nmodel = tf.keras.Sequential([tf.keras.layers.Dense(64)])",
                    "methods": {
                        "Model Creation": {
                            "Sequential": "model = tf.keras.Sequential()",
                            "Model": "model = tf.keras.Model(inputs=inputs, outputs=outputs)",
                            "Dense": "tf.keras.layers.Dense(64, activation='relu')",
                            "Conv2D": "tf.keras.layers.Conv2D(32, (3, 3), activation='relu')",
                            "LSTM": "tf.keras.layers.LSTM(50)"
                        },
                        "Training": {
                            "model.compile()": "model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])",
                            "model.fit()": "model.fit(X_train, y_train, epochs=10, batch_size=32)",
                            "model.evaluate()": "loss, accuracy = model.evaluate(X_test, y_test)",
                            "model.predict()": "predictions = model.predict(X_test)"
                        },
                        "Callbacks": {
                            "EarlyStopping": "tf.keras.callbacks.EarlyStopping(patience=5)",
                            "ModelCheckpoint": "tf.keras.callbacks.ModelCheckpoint('model.h5')",
                            "ReduceLROnPlateau": "tf.keras.callbacks.ReduceLROnPlateau(factor=0.5)"
                        }
                    }
                },
                "pytorch": {
                    "desc": "PyTorch deep learning framework",
                    "example": "import torch\nimport torch.nn as nn\nmodel = nn.Linear(10, 1)",
                    "methods": {
                        "Tensor Operations": {
                            "torch.tensor()": "x = torch.tensor([1, 2, 3], dtype=torch.float32)",
                            "torch.zeros()": "x = torch.zeros(3, 4)",
                            "torch.ones()": "x = torch.ones(2, 3)",
                            "torch.randn()": "x = torch.randn(2, 3)  # Normal distribution",
                            "torch.rand()": "x = torch.rand(2, 3)   # Uniform [0,1]",
                            "tensor.cuda()": "x = x.cuda()  # Move to GPU",
                            "tensor.cpu()": "x = x.cpu()   # Move to CPU"
                        },
                        "Neural Network Layers": {
                            "nn.Linear()": "layer = nn.Linear(in_features=10, out_features=1)",
                            "nn.Conv2d()": "conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)",
                            "nn.LSTM()": "lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2)",
                            "nn.Dropout()": "dropout = nn.Dropout(p=0.5)",
                            "nn.BatchNorm2d()": "bn = nn.BatchNorm2d(num_features=64)",
                            "nn.ReLU()": "relu = nn.ReLU()",
                            "nn.Softmax()": "softmax = nn.Softmax(dim=1)"
                        },
                        "Model Definition": {
                            "nn.Module": "class MyModel(nn.Module):\n    def __init__(self):\n        super().__init__()",
                            "nn.Sequential": "model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))",
                            "forward()": "def forward(self, x):\n    return self.layer(x)"
                        },
                        "Training": {
                            "torch.optim.Adam": "optimizer = torch.optim.Adam(model.parameters(), lr=0.001)",
                            "torch.optim.SGD": "optimizer = torch.optim.SGD(model.parameters(), lr=0.01)",
                            "nn.CrossEntropyLoss": "criterion = nn.CrossEntropyLoss()",
                            "nn.MSELoss": "criterion = nn.MSELoss()",
                            "loss.backward()": "loss.backward()  # Backpropagation",
                            "optimizer.step()": "optimizer.step()  # Update weights",
                            "optimizer.zero_grad()": "optimizer.zero_grad()  # Clear gradients"
                        },
                        "Data Loading": {
                            "DataLoader": "from torch.utils.data import DataLoader\nloader = DataLoader(dataset, batch_size=32)",
                            "Dataset": "from torch.utils.data import Dataset\nclass MyDataset(Dataset):",
                            "TensorDataset": "from torch.utils.data import TensorDataset\ndataset = TensorDataset(X, y)"
                        }
                    }
                },
                "lightgbm": {
                    "desc": "Gradient boosting framework by Microsoft",
                    "example": "import lightgbm as lgb\nmodel = lgb.LGBMClassifier().fit(X_train, y_train)",
                    "methods": {
                        "Models": {
                            "LGBMClassifier": "import lightgbm as lgb\nmodel = lgb.LGBMClassifier()",
                            "LGBMRegressor": "model = lgb.LGBMRegressor()",
                            "LGBMRanker": "model = lgb.LGBMRanker()"
                        },
                        "Training": {
                            "model.fit()": "model.fit(X_train, y_train, eval_set=[(X_val, y_val)])",
                            "model.predict()": "predictions = model.predict(X_test)",
                            "model.predict_proba()": "probabilities = model.predict_proba(X_test)"
                        },
                        "Feature Importance": {
                            "model.feature_importances_": "importances = model.feature_importances_",
                            "lgb.plot_importance()": "lgb.plot_importance(model, max_num_features=10)"
                        }
                    }
                },
                "catboost": {
                    "desc": "Gradient boosting framework by Yandex",
                    "example": "from catboost import CatBoostClassifier\nmodel = CatBoostClassifier().fit(X_train, y_train)",
                    "methods": {
                        "Models": {
                            "CatBoostClassifier": "from catboost import CatBoostClassifier\nmodel = CatBoostClassifier()",
                            "CatBoostRegressor": "from catboost import CatBoostRegressor\nmodel = CatBoostRegressor()",
                            "CatBoostRanker": "from catboost import CatBoostRanker\nmodel = CatBoostRanker()"
                        },
                        "Training": {
                            "model.fit()": "model.fit(X_train, y_train, eval_set=(X_val, y_val))",
                            "model.predict()": "predictions = model.predict(X_test)",
                            "model.predict_proba()": "probabilities = model.predict_proba(X_test)"
                        },
                        "Feature Importance": {
                            "model.feature_importances_": "importances = model.feature_importances_",
                            "model.get_feature_importance()": "importance = model.get_feature_importance(prettified=True)"
                        }
                    }
                }
            },
            "📈 Visualization": {
                "matplotlib": {
                    "desc": "Plotting and visualization library",
                    "example": "import matplotlib.pyplot as plt\nplt.plot([1,2,3], [4,5,6])\nplt.show()",
                    "methods": {
                        "Basic Plots": {
                            "plt.plot()": "plt.plot(x, y, 'b-', label='Line')",
                            "plt.scatter()": "plt.scatter(x, y, c='red', alpha=0.6)",
                            "plt.bar()": "plt.bar(categories, values)",
                            "plt.hist()": "plt.hist(data, bins=20, alpha=0.7)",
                            "plt.boxplot()": "plt.boxplot(data)",
                            "plt.pie()": "plt.pie(sizes, labels=labels, autopct='%1.1f%%')"
                        },
                        "Plot Customization": {
                            "plt.xlabel()": "plt.xlabel('X Axis Label')",
                            "plt.ylabel()": "plt.ylabel('Y Axis Label')",
                            "plt.title()": "plt.title('Plot Title')",
                            "plt.legend()": "plt.legend()",
                            "plt.grid()": "plt.grid(True, alpha=0.3)",
                            "plt.xlim()": "plt.xlim(0, 10)",
                            "plt.ylim()": "plt.ylim(0, 100)"
                        },
                        "Subplots": {
                            "plt.subplot()": "plt.subplot(2, 2, 1)  # 2x2 grid, position 1",
                            "plt.subplots()": "fig, axes = plt.subplots(2, 2, figsize=(10, 8))",
                            "plt.tight_layout()": "plt.tight_layout()  # Adjust spacing"
                        },
                        "Styling": {
                            "plt.style.use()": "plt.style.use('seaborn')",
                            "plt.figure()": "plt.figure(figsize=(10, 6))",
                            "plt.savefig()": "plt.savefig('plot.png', dpi=300, bbox_inches='tight')"
                        }
                    }
                },
                "seaborn": {
                    "desc": "Statistical data visualization",
                    "example": "import seaborn as sns\nsns.scatterplot(x='col1', y='col2', data=df)",
                    "methods": {
                        "Distribution Plots": {
                            "sns.histplot()": "sns.histplot(data=df, x='column')",
                            "sns.kdeplot()": "sns.kdeplot(data=df, x='column')",
                            "sns.boxplot()": "sns.boxplot(data=df, x='category', y='value')",
                            "sns.violinplot()": "sns.violinplot(data=df, x='category', y='value')",
                            "sns.distplot()": "sns.distplot(df['column'])"
                        },
                        "Relational Plots": {
                            "sns.scatterplot()": "sns.scatterplot(data=df, x='x', y='y', hue='category')",
                            "sns.lineplot()": "sns.lineplot(data=df, x='x', y='y')",
                            "sns.relplot()": "sns.relplot(data=df, x='x', y='y', kind='scatter')"
                        },
                        "Categorical Plots": {
                            "sns.barplot()": "sns.barplot(data=df, x='category', y='value')",
                            "sns.countplot()": "sns.countplot(data=df, x='category')",
                            "sns.pointplot()": "sns.pointplot(data=df, x='category', y='value')"
                        },
                        "Matrix Plots": {
                            "sns.heatmap()": "sns.heatmap(correlation_matrix, annot=True)",
                            "sns.clustermap()": "sns.clustermap(data)",
                            "sns.pairplot()": "sns.pairplot(df, hue='target')"
                        },
                        "Styling": {
                            "sns.set_style()": "sns.set_style('whitegrid')",
                            "sns.set_palette()": "sns.set_palette('husl')",
                            "sns.color_palette()": "colors = sns.color_palette('viridis', 10)"
                        }
                    }
                },
                "plotly": {
                    "desc": "Interactive plotting library",
                    "example": "import plotly.express as px\nfig = px.scatter(df, x='x', y='y')\nfig.show()",
                    "methods": {
                        "Express Plots": {
                            "px.scatter()": "fig = px.scatter(df, x='x', y='y', color='category')",
                            "px.line()": "fig = px.line(df, x='x', y='y')",
                            "px.bar()": "fig = px.bar(df, x='category', y='value')",
                            "px.histogram()": "fig = px.histogram(df, x='column')",
                            "px.box()": "fig = px.box(df, x='category', y='value')"
                        },
                        "3D Plots": {
                            "px.scatter_3d()": "fig = px.scatter_3d(df, x='x', y='y', z='z')",
                            "px.line_3d()": "fig = px.line_3d(df, x='x', y='y', z='z')"
                        },
                        "Statistical Plots": {
                            "px.violin()": "fig = px.violin(df, x='category', y='value')",
                            "px.density_heatmap()": "fig = px.density_heatmap(df, x='x', y='y')",
                            "px.parallel_coordinates()": "fig = px.parallel_coordinates(df)"
                        },
                        "Customization": {
                            "fig.update_layout()": "fig.update_layout(title='My Plot')",
                            "fig.update_traces()": "fig.update_traces(marker_size=10)",
                            "fig.show()": "fig.show()"
                        }
                    }
                }
            },
            "🔬 Scientific Computing": {
                "scipy": {
                    "desc": "Scientific computing library",
                    "example": "from scipy import stats\nresult = stats.ttest_1samp(data, 0)",
                    "methods": {
                        "Statistics": {
                            "stats.ttest_1samp()": "from scipy import stats\nstats.ttest_1samp(data, popmean)",
                            "stats.ttest_ind()": "stats.ttest_ind(group1, group2)",
                            "stats.chi2_contingency()": "stats.chi2_contingency(contingency_table)",
                            "stats.pearsonr()": "stats.pearsonr(x, y)",
                            "stats.spearmanr()": "stats.spearmanr(x, y)"
                        },
                        "Optimization": {
                            "optimize.minimize()": "from scipy import optimize\noptimize.minimize(func, x0)",
                            "optimize.curve_fit()": "optimize.curve_fit(func, xdata, ydata)",
                            "optimize.fsolve()": "optimize.fsolve(func, x0)"
                        },
                        "Linear Algebra": {
                            "linalg.solve()": "from scipy import linalg\nlinalg.solve(A, b)",
                            "linalg.eig()": "linalg.eig(matrix)",
                            "linalg.svd()": "linalg.svd(matrix)"
                        }
                    }
                }
            },
            "🌐 Data Collection": {
                "requests": {
                    "desc": "HTTP library for API calls",
                    "example": "import requests\nresponse = requests.get('https://api.example.com')",
                    "methods": {
                        "HTTP Methods": {
                            "requests.get()": "response = requests.get('https://api.example.com')",
                            "requests.post()": "response = requests.post('https://api.example.com', data=data)",
                            "requests.put()": "response = requests.put('https://api.example.com', data=data)",
                            "requests.delete()": "response = requests.delete('https://api.example.com')"
                        },
                        "Response Handling": {
                            "response.json()": "data = response.json()",
                            "response.text": "text = response.text",
                            "response.status_code": "status = response.status_code",
                            "response.headers": "headers = response.headers"
                        }
                    }
                },
                "beautifulsoup4": {
                    "desc": "Web scraping library",
                    "example": "from bs4 import BeautifulSoup\nsoup = BeautifulSoup(html, 'html.parser')",
                    "methods": {
                        "Parsing": {
                            "BeautifulSoup()": "from bs4 import BeautifulSoup\nsoup = BeautifulSoup(html, 'html.parser')",
                            "soup.find()": "element = soup.find('div', class_='content')",
                            "soup.find_all()": "elements = soup.find_all('a')",
                            "soup.select()": "elements = soup.select('.class-name')"
                        },
                        "Data Extraction": {
                            "element.text": "text = element.text",
                            "element.get()": "href = element.get('href')",
                            "element.attrs": "attributes = element.attrs"
                        }
                    }
                }
            },
            "🎯 Computer Vision": {
                "ultralytics": {
                    "desc": "YOLO object detection and segmentation",
                    "example": "from ultralytics import YOLO\nmodel = YOLO('yolov8n.pt')",
                    "methods": {
                        "Model Loading": {
                            "YOLO()": "from ultralytics import YOLO\nmodel = YOLO('yolov8n.pt')",
                            "YOLO.load()": "model = YOLO.load('path/to/model.pt')",
                            "model.info()": "model.info()  # Model information"
                        },
                        "Detection": {
                            "model.predict()": "results = model.predict('image.jpg', save=True)",
                            "model()": "results = model('image.jpg')  # Same as predict",
                            "model.track()": "results = model.track('video.mp4', save=True)"
                        },
                        "Training": {
                            "model.train()": "model.train(data='coco128.yaml', epochs=100, imgsz=640)",
                            "model.val()": "metrics = model.val()  # Validation",
                            "model.export()": "model.export(format='onnx')  # Export model"
                        },
                        "Results Processing": {
                            "results.boxes": "boxes = results[0].boxes  # Bounding boxes",
                            "results.masks": "masks = results[0].masks  # Segmentation masks",
                            "results.keypoints": "keypoints = results[0].keypoints  # Pose keypoints",
                            "results.save()": "results[0].save('output.jpg')  # Save results",
                            "results.show()": "results[0].show()  # Display results"
                        }
                    }
                },
                "opencv": {
                    "desc": "Computer vision library",
                    "example": "import cv2\nimg = cv2.imread('image.jpg')",
                    "methods": {
                        "Image I/O": {
                            "cv2.imread()": "img = cv2.imread('image.jpg')",
                            "cv2.imwrite()": "cv2.imwrite('output.jpg', img)",
                            "cv2.imshow()": "cv2.imshow('window', img); cv2.waitKey(0)",
                            "cv2.destroyAllWindows()": "cv2.destroyAllWindows()"
                        },
                        "Image Processing": {
                            "cv2.resize()": "resized = cv2.resize(img, (width, height))",
                            "cv2.cvtColor()": "gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)",
                            "cv2.GaussianBlur()": "blurred = cv2.GaussianBlur(img, (15, 15), 0)",
                            "cv2.Canny()": "edges = cv2.Canny(gray, 50, 150)",
                            "cv2.threshold()": "_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)"
                        },
                        "Feature Detection": {
                            "cv2.goodFeaturesToTrack()": "corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)",
                            "cv2.SIFT_create()": "sift = cv2.SIFT_create()\nkp, des = sift.detectAndCompute(gray, None)",
                            "cv2.ORB_create()": "orb = cv2.ORB_create()\nkp, des = orb.detectAndCompute(gray, None)"
                        }
                    }
                }
            }
        }
        
        # Create expandable sections
        for category, libs in libraries.items():
            # Category header
            cat_frame = tk.Frame(scrollable_frame, bg='#3c3c3c')
            cat_frame.pack(fill=tk.X, pady=2)
            
            cat_label = tk.Label(cat_frame, text=category, 
                               bg='#3c3c3c', fg='#ffffff',
                               font=('Consolas', 10, 'bold'))
            cat_label.pack(pady=5)
            
            # Libraries in category
            for lib_name, lib_info in libs.items():
                lib_frame_item = tk.Frame(scrollable_frame, bg='#404040', relief='raised', bd=1)
                lib_frame_item.pack(fill=tk.X, pady=1, padx=5)
                
                # Header frame với expand button
                header_frame = tk.Frame(lib_frame_item, bg='#404040')
                header_frame.pack(fill=tk.X, padx=5, pady=2)
                
                # Expand/Collapse button
                expand_var = tk.BooleanVar()
                expand_btn = tk.Button(header_frame, text="▶", font=('Arial', 8), 
                                     bg='#4CAF50', fg='white', width=2)
                expand_btn.pack(side='left', padx=(0, 5))
                
                # Library info frame
                info_frame = tk.Frame(header_frame, bg='#404040')
                info_frame.pack(side='left', fill='x', expand=True)
                
                # Library name and description
                lib_label = tk.Label(info_frame, text=f"• {lib_name}", 
                                   bg='#404040', fg='#4CAF50',
                                   font=('Consolas', 9, 'bold'),
                                   anchor='w')
                lib_label.pack(fill=tk.X)
                
                desc_label = tk.Label(info_frame, text=lib_info['desc'], 
                                    bg='#404040', fg='#cccccc',
                                    font=('Consolas', 8),
                                    anchor='w', wraplength=180)
                desc_label.pack(fill=tk.X, padx=(10, 0))
                
                # Insert example button
                def insert_example(example=lib_info['example']):
                    self.code_text.insert(tk.END, f"\n# {lib_name} example:\n{example}\n\n")
                    self.code_text.see(tk.END)
                
                example_btn = tk.Button(header_frame, text="📝 Insert Example",
                                      command=insert_example,
                                      bg='#4CAF50', fg='white',
                                      font=('Consolas', 8),
                                      relief=tk.FLAT)
                example_btn.pack(side='right', padx=5)
                
                # Methods container (initially hidden)
                methods_frame = tk.Frame(lib_frame_item, bg='#4d4d4d')
                # Don't pack initially - will be shown/hidden by toggle
                
                # Add methods if they exist
                if 'methods' in lib_info:
                    for method_category, methods in lib_info['methods'].items():
                        # Method category header
                        method_cat_frame = tk.Frame(methods_frame, bg='#4d4d4d')
                        method_cat_frame.pack(fill='x', padx=10, pady=2)
                        
                        method_cat_label = tk.Label(method_cat_frame, text=f"• {method_category}", 
                                                  font=('Consolas', 9, 'bold'), bg='#4d4d4d', fg='#FFA726')
                        method_cat_label.pack(anchor='w')
                        
                        # Individual methods
                        for method_name, method_code in methods.items():
                            method_frame = tk.Frame(methods_frame, bg='#5d5d5d')
                            method_frame.pack(fill='x', padx=20, pady=1)
                            
                            method_label = tk.Label(method_frame, text=method_name, 
                                                  font=('Consolas', 8), bg='#5d5d5d', fg='#E0E0E0')
                            method_label.pack(side='left', anchor='w')
                            
                            def insert_method(code=method_code):
                                self.code_text.insert(tk.END, f"\n{code}\n")
                                self.code_text.see(tk.END)
                            
                            method_btn = tk.Button(method_frame, text="Insert", 
                                                 command=insert_method,
                                                 bg='#FF7043', fg='white', font=('Consolas', 7), width=6)
                            method_btn.pack(side='right', padx=2)
                
                # Configure expand button command
                def toggle_methods(frame=lib_frame_item, var=expand_var, btn=expand_btn, methods=methods_frame):
                    if var.get():
                        methods.pack_forget()
                        btn.config(text="▶")
                        var.set(False)
                    else:
                        methods.pack(fill='x', padx=5, pady=5)
                        btn.config(text="▼")
                        var.set(True)
                
                expand_btn.config(command=toggle_methods)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def create_code_editor(self, parent):
        """Create the code editor panel"""
        editor_frame = tk.Frame(parent, bg='#1e1e1e', relief=tk.RAISED, bd=2, width=400)
        editor_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.editor_frame = editor_frame
        
        tk.Label(editor_frame, text="💻 Python ML Code Editor", bg='#1e1e1e', fg='#4CAF50',
                font=('Arial', 14, 'bold')).pack(pady=5)
        
        # Code editor with line numbers
        editor_container = tk.Frame(editor_frame, bg='#1e1e1e')
        editor_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Line numbers
        self.line_numbers = tk.Text(editor_container, width=4, bg='#2d2d2d', fg='#888888',
                                   font=('Consolas', 11), state=tk.DISABLED, wrap=tk.NONE)
        self.line_numbers.pack(side=tk.LEFT, fill=tk.Y)
        
        # Code text area
        self.code_text = tk.Text(editor_container, bg='#1e1e1e', fg='#ffffff',
                                font=('Consolas', 11), wrap=tk.NONE, insertbackground='white')
        
        # Scrollbars
        v_scrollbar = tk.Scrollbar(editor_container, orient=tk.VERTICAL)
        h_scrollbar = tk.Scrollbar(editor_frame, orient=tk.HORIZONTAL)
        
        # Configure scrolling
        self.code_text.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        v_scrollbar.configure(command=self.sync_scroll)
        h_scrollbar.configure(command=self.code_text.xview)
        
        # Pack scrollbars and text
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.code_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        h_scrollbar.pack(fill=tk.X)
        
        # Bind events
        self.code_text.bind('<KeyRelease>', self.on_code_change)
        self.code_text.bind('<Button-1>', self.on_code_change)
        
        # Configure syntax highlighting tags
        self.setup_syntax_highlighting()
        
        # Update line numbers initially
        self.update_line_numbers()
    
    def create_block_canvas(self, parent):
        """Create the visualization panel with toggleable views (Blocks / 3D)"""
        self.canvas_frame = tk.Frame(parent, bg='#1e1e1e', relief=tk.RAISED, bd=2, width=500)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.canvas_frame.pack_propagate(False)

        self.visual_title = tk.Label(self.canvas_frame, text="🧩 Live Block Visualization", bg='#1e1e1e', fg='#4CAF50',
                                     font=('Arial', 14, 'bold'))
        self.visual_title.pack(pady=5)

        # View: Blocks (default)
        self.block_view_frame = tk.Frame(self.canvas_frame, bg='#1e1e1e')
        self.block_view_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        canvas_container = tk.Frame(self.block_view_frame, bg='#1e1e1e')
        canvas_container.pack(fill=tk.BOTH, expand=True)

        self.block_canvas = tk.Canvas(canvas_container, bg='#2b2b2b', scrollregion=(0, 0, 1000, 2000))
        canvas_v_scroll = tk.Scrollbar(canvas_container, orient=tk.VERTICAL, command=self.block_canvas.yview)
        canvas_h_scroll = tk.Scrollbar(canvas_container, orient=tk.HORIZONTAL, command=self.block_canvas.xview)
        self.block_canvas.configure(yscrollcommand=canvas_v_scroll.set, xscrollcommand=canvas_h_scroll.set)
        canvas_v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas_h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.block_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 3D view disabled; no three_d_view_frame created
        
        # Draw grid
        self.draw_canvas_grid()
        # Canvas interactions: zoom/pan
        self.block_canvas.bind("<Control-MouseWheel>", self.on_canvas_zoom)
        self.block_canvas.bind("<Button-2>", self.on_pan_start)
        self.block_canvas.bind("<B2-Motion>", self.on_pan_move)
        self.block_canvas.bind("<ButtonRelease-2>", self.on_pan_end)
        self.block_canvas.bind("<Shift-Button-1>", self.on_pan_start)
        self.block_canvas.bind("<Shift-B1-Motion>", self.on_pan_move)
        self.block_canvas.bind("<Shift-ButtonRelease-1>", self.on_pan_end)
    
    def create_output_panel(self, parent):
        """Create the output and visualization panel"""
        output_frame = tk.Frame(parent, bg='#1e1e1e', relief=tk.RAISED, bd=2, width=400)
        output_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        output_frame.pack_propagate(False)
        self.output_frame = output_frame
        
        # Notebook for tabs
        notebook = ttk.Notebook(output_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.output_notebook = notebook
        
        # Visualization tab
        viz_tab = tk.Frame(notebook, bg='#1e1e1e')
        notebook.add(viz_tab, text="📊 Plot")
        
        # Matplotlib figure
        self.fig = Figure(figsize=(5, 4), dpi=80, facecolor='#2b2b2b')
        self.ax = self.fig.add_subplot(111, facecolor='#2b2b2b')
        self.ax.tick_params(colors='white')
        
        self.plot_canvas = FigureCanvasTkAgg(self.fig, viz_tab)
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Variables tab
        vars_tab = tk.Frame(notebook, bg='#1e1e1e')
        notebook.add(vars_tab, text="📋 Variables")
        
        self.vars_text = tk.Text(vars_tab, bg='#1e1e1e', fg='white',
                                font=('Consolas', 10), wrap=tk.WORD)
        vars_scrollbar = tk.Scrollbar(vars_tab, command=self.vars_text.yview)
        self.vars_text.configure(yscrollcommand=vars_scrollbar.set)
        
        vars_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.vars_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Help tab
        help_tab = tk.Frame(notebook, bg='#1e1e1e')
        notebook.add(help_tab, text="❓ Help")
        
        self.help_text = tk.Text(help_tab, bg='#1e1e1e', fg='white',
                                font=('Consolas', 9), wrap=tk.WORD)
        help_scrollbar = tk.Scrollbar(help_tab, command=self.help_text.yview)
        self.help_text.configure(yscrollcommand=help_scrollbar.set)
        
        help_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.help_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.load_help_content()

        # Layers tab - CNN-style animation
        layers_tab = tk.Frame(notebook, bg='#1e1e1e')
        notebook.add(layers_tab, text="🧠 Layers")
        self.layers_tab = layers_tab
        self.init_layers_tab(layers_tab)
    
    def create_console_panel(self, parent):
        """Create the console panel"""
        console_frame = tk.Frame(parent, bg='#1e1e1e', relief=tk.RAISED, bd=2, height=200)
        console_frame.pack(fill=tk.X, pady=(10, 0))
        console_frame.pack_propagate(False)
        self.console_frame = console_frame
        
        tk.Label(console_frame, text="🖥️ Console Output", bg='#1e1e1e', fg='#4CAF50',
                font=('Arial', 12, 'bold')).pack(pady=5)
        
        # Console text area
        console_container = tk.Frame(console_frame, bg='#1e1e1e')
        console_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.console_text = tk.Text(console_container, bg='#000000', fg='#00ff00',
                                   font=('Consolas', 10), height=10)
        console_scroll = tk.Scrollbar(console_container, command=self.console_text.yview)
        self.console_text.configure(yscrollcommand=console_scroll.set)
        
        console_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.console_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure console tags
        self.console_text.tag_configure("output", foreground="#00ff00")
        self.console_text.tag_configure("error", foreground="#ff4444")
        self.console_text.tag_configure("info", foreground="#4CAF50")
    
    def setup_syntax_highlighting(self):
        """Setup syntax highlighting for Python code"""
        # Define color tags
        self.code_text.tag_configure("keyword", foreground="#569CD6")
        self.code_text.tag_configure("string", foreground="#CE9178")
        self.code_text.tag_configure("comment", foreground="#6A9955")
        self.code_text.tag_configure("number", foreground="#B5CEA8")
        self.code_text.tag_configure("function", foreground="#DCDCAA")
        self.code_text.tag_configure("current_line", background="#2d4f67")

    def setup_keybindings(self):
        """Bind developer-friendly keyboard shortcuts"""
        # Global shortcuts
        self.root.bind('<F5>', lambda e: self.run_code())
        self.root.bind('<F10>', lambda e: self.run_step_by_step())
        self.root.bind('<Shift-F5>', lambda e: self.stop_execution())
        self.root.bind('<F7>', lambda e: self.open_layers_tab_and_run())

    # ==== Layers (Conv → ReLU → Pool) animation ====
    def init_layers_tab(self, parent):
        """Initialize Layers tab UI: draw pad and 2x2 plots."""
        container = tk.Frame(parent, bg='#1e1e1e')
        container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left = tk.Frame(container, bg='#1e1e1e')
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        right = tk.Frame(container, bg='#1e1e1e')
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(left, text="✍️ Draw (28×28)", bg='#1e1e1e', fg='#4CAF50',
                 font=('Arial', 12, 'bold')).pack(pady=(0, 5))

        cell = 10
        pad_size = self.layers_grid_size * cell
        canvas = tk.Canvas(left, width=pad_size, height=pad_size, bg='black', highlightthickness=1, highlightbackground='#4a4a4a')
        canvas.pack(pady=5)
        self.layers_pad_canvas = canvas

        # Draw grid
        for i in range(self.layers_grid_size + 1):
            x = i * cell
            canvas.create_line(x, 0, x, pad_size, fill='#333333')
            canvas.create_line(0, x, pad_size, x, fill='#333333')

        # Bind drawing
        canvas.bind('<Button-1>', lambda e: self.on_pad_draw(e, cell))
        canvas.bind('<B1-Motion>', lambda e: self.on_pad_draw(e, cell))

        # Controls
        ctrl = tk.Frame(left, bg='#1e1e1e')
        ctrl.pack(fill=tk.X, pady=5)
        tk.Button(ctrl, text="🧹 Clear", command=self.clear_layers_pad,
                 bg='#607D8B', fg='white').pack(side=tk.LEFT, padx=3)
        tk.Button(ctrl, text="▶ Predict", command=self.run_layers_predict,
                 bg='#4CAF50', fg='white').pack(side=tk.LEFT, padx=3)
        tk.Label(ctrl, text="Speed (ms)", bg='#1e1e1e', fg='white').pack(side=tk.LEFT, padx=(10, 3))
        tk.Scale(ctrl, from_=50, to=400, orient=tk.HORIZONTAL, variable=self.layers_speed_var,
                 bg='#1e1e1e', fg='white', length=160, command=lambda v: self.set_layers_speed(int(float(v))))\
            .pack(side=tk.LEFT, padx=3)

        # 2x2 plots
        self.layers_fig = Figure(figsize=(6.2, 5.4), dpi=80, facecolor='#2b2b2b')
        # 2x2 trên cùng
        ax1 = self.layers_fig.add_subplot(231)
        ax2 = self.layers_fig.add_subplot(232)
        ax3 = self.layers_fig.add_subplot(234)
        ax4 = self.layers_fig.add_subplot(235)
        # Ô phải dưới: xác suất 10 lớp
        ax5 = self.layers_fig.add_subplot(233)
        # Ô trái dưới: vector phẳng
        ax6 = self.layers_fig.add_subplot(236)
        for ax in (ax1, ax2, ax3, ax4):
            ax.set_facecolor('#2b2b2b')
            ax.tick_params(colors='white')
        ax1.set_title('Input', color='white', fontsize=10)
        ax2.set_title('Convolution', color='white', fontsize=10)
        ax3.set_title('ReLU', color='white', fontsize=10)
        ax4.set_title('Pooling 2×2', color='white', fontsize=10)
        ax5.set_title('Output Prob (0-9)', color='white', fontsize=10)
        ax6.set_title('Flatten', color='white', fontsize=10)
        self.layers_axes = {'input': ax1, 'conv': ax2, 'relu': ax3, 'pool': ax4, 'out': ax5, 'flat': ax6}

        self.layers_plot_canvas = FigureCanvasTkAgg(self.layers_fig, right)
        self.layers_plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.update_layers_plots(reset=True)

    def set_layers_speed(self, ms):
        self.layers_anim_speed_ms = int(ms)

    def clear_layers_pad(self):
        self.layers_img[:] = 0
        self.layers_pad_canvas.delete('paint')
        # Redraw grid borders remain; repaint background
        self.layers_pad_canvas.configure(bg='black')
        self.update_layers_plots(reset=True)

    def on_pad_draw(self, event, cell):
        # Map mouse to grid cell
        x, y = event.x, event.y
        r = max(0, min(self.layers_grid_size - 1, y // cell))
        c = max(0, min(self.layers_grid_size - 1, x // cell))
        self.layers_img[r, c] = 1.0
        # Paint the cell
        x0, y0 = c * cell, r * cell
        x1, y1 = x0 + cell, y0 + cell
        self.layers_pad_canvas.create_rectangle(x0, y0, x1, y1, fill='white', outline='white', tags='paint')

    def normalize(self, arr):
        if arr is None:
            return None
        if np.all(arr == 0):
            return arr
        m, M = float(arr.min()), float(arr.max())
        if M - m < 1e-9:
            return np.zeros_like(arr)
        return (arr - m) / (M - m)

    def update_layers_plots(self, reset=False):
        # Show current arrays
        ax1, ax2, ax3, ax4 = self.layers_axes['input'], self.layers_axes['conv'], self.layers_axes['relu'], self.layers_axes['pool']
        ax_out, ax_flat = self.layers_axes['out'], self.layers_axes['flat']
        ax1.clear(); ax1.set_facecolor('#2b2b2b'); ax1.set_title('Input', color='white', fontsize=10)
        ax2.clear(); ax2.set_facecolor('#2b2b2b'); ax2.set_title('Convolution', color='white', fontsize=10)
        ax3.clear(); ax3.set_facecolor('#2b2b2b'); ax3.set_title('ReLU', color='white', fontsize=10)
        ax4.clear(); ax4.set_facecolor('#2b2b2b'); ax4.set_title('Pooling 2×2', color='white', fontsize=10)
        ax_out.clear(); ax_out.set_facecolor('#2b2b2b'); ax_out.set_title('Output Prob (0-9)', color='white', fontsize=10)
        ax_flat.clear(); ax_flat.set_facecolor('#2b2b2b'); ax_flat.set_title('Flatten', color='white', fontsize=10)

        ax1.imshow(self.layers_img, cmap='gray', vmin=0, vmax=1)
        if reset:
            self.layers_conv_map = np.zeros((self.layers_img.shape[0]-2, self.layers_img.shape[1]-2))
            self.layers_relu_map = np.zeros_like(self.layers_conv_map)
            self.layers_pool_map = np.zeros((self.layers_relu_map.shape[0]//2, self.layers_relu_map.shape[1]//2))
            self.layers_flat = None
            self.layers_logits = None
            self.layers_probs = None
        ax2.imshow(self.normalize(self.layers_conv_map), cmap='gray', vmin=0, vmax=1)
        ax3.imshow(self.normalize(self.layers_relu_map), cmap='gray', vmin=0, vmax=1)
        ax4.imshow(self.normalize(self.layers_pool_map), cmap='gray', vmin=0, vmax=1)

        # Flatten view
        if self.layers_flat is not None:
            flat_vis = self.normalize(self.layers_flat)
            ax_flat.imshow(flat_vis.reshape(1, -1), cmap='gray', aspect='auto', vmin=0, vmax=1)
            ax_flat.get_yaxis().set_visible(False)
            ax_flat.get_xaxis().set_visible(False)
        else:
            ax_flat.text(0.5, 0.5, 'Waiting for pool → flatten', color='white', ha='center', va='center', fontsize=9)

        # Output probabilities
        if self.layers_probs is not None:
            digits = list(range(10))
            ax_out.bar(digits, self.layers_probs, color='#4CAF50')
            ax_out.set_xticks(digits)
            ax_out.set_xticklabels([str(d) for d in digits], color='white')
            ax_out.tick_params(axis='y', colors='white')
            ax_out.set_ylim(0, 1)
        else:
            ax_out.text(0.5, 0.5, 'Waiting for dense → softmax', color='white', ha='center', va='center', fontsize=9)

        self.layers_fig.tight_layout()
        self.layers_plot_canvas.draw()

    def conv2d_valid(self, img, kernel):
        h, w = img.shape
        kh, kw = kernel.shape
        out = np.zeros((h - kh + 1, w - kw + 1))
        for i in range(h - kh + 1):
            for j in range(w - kw + 1):
                region = img[i:i+kh, j:j+kw]
                out[i, j] = float(np.sum(region * kernel))
        return out

    def max_pool_2x2(self, img):
        h, w = img.shape
        ph, pw = h // 2, w // 2
        out = np.zeros((ph, pw))
        for i in range(ph):
            for j in range(pw):
                region = img[2*i:2*i+2, 2*j:2*j+2]
                out[i, j] = float(np.max(region))
        return out

    def open_layers_tab_and_run(self):
        if self.layers_tab is not None:
            try:
                self.output_notebook.select(self.layers_tab)
            except Exception:
                pass
        self.run_layers_predict()

    def run_layers_predict(self):
        # Prepare scanning sequence and arrays
        self.layers_anim_running = True
        self.layers_conv_map = np.zeros((self.layers_img.shape[0]-2, self.layers_img.shape[1]-2))
        self.layers_relu_map = np.zeros_like(self.layers_conv_map)
        self.layers_pool_map = np.zeros((self.layers_relu_map.shape[0]//2, self.layers_relu_map.shape[1]//2))

        # Positions to scan (top-left of 3x3 windows)
        h, w = self.layers_img.shape
        self.layers_scan_positions = [(i, j) for i in range(h-2) for j in range(w-2)]
        self.layers_scan_index = 0

        # Add or reset scanning rectangle on input
        ax1 = self.layers_axes['input']
        if self.layers_input_rect is not None:
            try:
                self.layers_input_rect.remove()
            except Exception:
                pass
        self.layers_input_rect = Rectangle((0, 0), 3, 3, linewidth=1.2, edgecolor='cyan', facecolor='none')
        ax1.add_patch(self.layers_input_rect)

        self.update_layers_plots(reset=True)
        self.step_layers_animation()

    def step_layers_animation(self):
        if not self.layers_anim_running:
            return
        if self.layers_scan_index < len(self.layers_scan_positions):
            i, j = self.layers_scan_positions[self.layers_scan_index]
            # Compute one conv cell
            val = float(np.sum(self.layers_img[i:i+3, j:j+3] * self.layers_conv_kernel))
            self.layers_conv_map[i, j] = val
            # Move rectangle
            try:
                self.layers_input_rect.set_xy((j, i))
            except Exception:
                pass
            # Update plots
            self.update_layers_plots(reset=False)
            self.layers_scan_index += 1
            self.root.after(self.layers_anim_speed_ms, self.step_layers_animation)
        else:
            # Finish: ReLU and Pool
            self.layers_relu_map = np.maximum(self.layers_conv_map, 0)
            self.layers_pool_map = self.max_pool_2x2(self.layers_relu_map)
            self.update_layers_plots(reset=False)
            self.layers_anim_running = False
            # Proceed to dense + softmax visualization
            self.run_dense_and_visualize()

    def softmax(self, x):
        x = np.asarray(x, dtype=float)
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    def run_dense_and_visualize(self):
        # Flatten after pooling
        if self.layers_pool_map is None:
            return
        self.layers_flat = self.layers_pool_map.flatten()
        n_features = self.layers_flat.size
        # Init weights lazily and deterministically
        if (self.layers_dense_W is None) or (self.layers_dense_W.shape[0] != n_features):
            rng = np.random.default_rng(42)
            self.layers_dense_W = rng.normal(0, 0.2, size=(n_features, 10))
            self.layers_dense_b = rng.normal(0, 0.05, size=(10,))
        # Compute logits and probs
        self.layers_logits = self.layers_flat @ self.layers_dense_W + self.layers_dense_b
        self.layers_probs = self.softmax(self.layers_logits)
        # Simple reveal animation of bars
        self.layers_dense_step = 0
        self.reveal_prob_bars_step()

    def reveal_prob_bars_step(self):
        if self.layers_probs is None:
            return
        # Progressive display: set probs after step passes index
        step = self.layers_dense_step
        probs = np.zeros_like(self.layers_probs)
        if step >= 10:
            probs = self.layers_probs
        else:
            probs[:step] = self.layers_probs[:step]
        # Temporarily assign and draw
        prev = self.layers_probs
        self.layers_probs = probs
        self.update_layers_plots(reset=False)
        # Restore full probs after drawing state, then schedule next
        self.layers_probs = prev
        if step < 10:
            self.layers_dense_step += 1
            self.root.after(self.layers_anim_speed_ms, self.reveal_prob_bars_step)
        self.root.bind('<F9>', lambda e: self.toggle_dev_focus())
        self.root.bind('<Control-s>', lambda e: self.save_code())
        self.root.bind('<Control-o>', lambda e: self.load_code())
        # Editor-specific
        if hasattr(self, 'code_text'):
            self.code_text.bind('<Control-Return>', lambda e: self.run_selection_or_line())

    def toggle_dev_focus(self):
        """Toggle Dev Focus mode: prioritize coding by hiding non-essential panels"""
        self.dev_focus = not self.dev_focus
        if self.dev_focus:
            # Hide auxiliary panels
            try:
                self.lib_frame.pack_forget()
            except Exception:
                pass
            try:
                self.canvas_frame.pack_forget()
            except Exception:
                pass
            try:
                self.output_frame.pack_forget()
            except Exception:
                pass
            # Ensure editor expands fully
            self.editor_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
            # Update button
            self.dev_focus_btn.configure(text="🎯 Dev Focus: ON", bg='#4CAF50')
        else:
            # Restore panels
            self.lib_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
            self.editor_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
            self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            self.output_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
            self.dev_focus_btn.configure(text="🎯 Dev Focus: OFF", bg='#3E3E3E')

    def run_selection_or_line(self):
        """Run selected code or the current line"""
        if self.is_running:
            return
        # Determine selection
        try:
            sel_ranges = self.code_text.tag_ranges("sel")
            if sel_ranges:
                snippet = self.code_text.get(sel_ranges[0], sel_ranges[1])
            else:
                line_start = self.code_text.index("insert linestart")
                line_end = self.code_text.index("insert lineend")
                snippet = self.code_text.get(line_start, line_end)
        except Exception:
            return

        if not snippet.strip():
            return

        self.is_running = True
        self.console_text.insert(tk.END, "⚡ Run snippet...\n", "info")
        threading.Thread(target=lambda: self.execute_snippet(snippet), daemon=True).start()

    def execute_snippet(self, snippet):
        """Execute a code snippet (selection or single line)"""
        try:
            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            # Execute snippet in the same globals
            exec(snippet, self.execution_globals)
            # Get output
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            if output:
                self.root.after(0, lambda: self.console_text.insert(tk.END, output, "output"))
            # Update variables and plot
            self.root.after(0, self.update_variables_display)
            self.root.after(0, self.update_plot_display)
        except Exception as e:
            self.root.after(0, lambda: self.console_text.insert(tk.END, f"❌ Error: {str(e)}\n", "error"))
        finally:
            self.is_running = False
    
    def sync_scroll(self, *args):
        """Synchronize scrolling between code text and line numbers"""
        self.line_numbers.yview_moveto(args[0])
        self.code_text.yview_moveto(args[0])
    
    def update_line_numbers(self):
        """Update line numbers"""
        self.line_numbers.config(state=tk.NORMAL)
        self.line_numbers.delete(1.0, tk.END)
        
        code_content = self.code_text.get(1.0, tk.END)
        lines = code_content.split('\n')
        
        for i in range(1, len(lines)):
            self.line_numbers.insert(tk.END, f"{i:3d}\n")
        
        self.line_numbers.config(state=tk.DISABLED)
    
    def on_code_change(self, event=None):
        """Handle code changes"""
        self.update_line_numbers()
        self.apply_syntax_highlighting()
        self.analyze_code_structure()
    
    def apply_syntax_highlighting(self):
        """Apply syntax highlighting to the code"""
        # Clear existing tags
        for tag in ["keyword", "string", "comment", "number", "function"]:
            self.code_text.tag_remove(tag, 1.0, tk.END)
        
        code = self.code_text.get(1.0, tk.END)
        
        # Keywords
        keywords = ['import', 'from', 'def', 'class', 'if', 'else', 'elif', 'for', 'while', 
                   'try', 'except', 'finally', 'with', 'as', 'return', 'yield', 'lambda',
                   'and', 'or', 'not', 'in', 'is', 'True', 'False', 'None']
        
        for keyword in keywords:
            self.highlight_pattern(rf'\b{keyword}\b', "keyword")
        
        # Strings
        self.highlight_pattern(r'"[^"]*"', "string")
        self.highlight_pattern(r"'[^']*'", "string")
        
        # Comments
        self.highlight_pattern(r'#.*', "comment")
        
        # Numbers
        self.highlight_pattern(r'\b\d+\.?\d*\b', "number")
        
        # Functions
        self.highlight_pattern(r'\b\w+(?=\()', "function")
    
    def highlight_pattern(self, pattern, tag):
        """Highlight a specific pattern with a tag"""
        code = self.code_text.get(1.0, tk.END)
        
        for match in re.finditer(pattern, code):
            start_idx = f"1.0+{match.start()}c"
            end_idx = f"1.0+{match.end()}c"
            self.code_text.tag_add(tag, start_idx, end_idx)
    
    def analyze_code_structure(self):
        """Analyze code structure and update block visualization"""
        code = self.code_text.get(1.0, tk.END)
        lines = code.split('\n')
        
        # Clear existing blocks
        self.block_canvas.delete("all")
        self.draw_canvas_grid()
        self.code_blocks.clear()
        self.block_tag_map.clear()
        
        # Column layout by code type
        categories_order = [
            'import', 'data_load', 'preprocessing', 'model',
            'training', 'prediction', 'evaluation', 'visualization'
        ]
        col_x = {cat: 50 + idx * 170 for idx, cat in enumerate(categories_order)}
        col_count = {cat: 0 for cat in categories_order}
        y_offset = 50
        i = 1
        n = len(lines)
        while i <= n:
            line_stripped = lines[i-1].strip()
            if not line_stripped or line_stripped.startswith('#'):
                i += 1
                continue
            code_type = self.classify_code_line(line_stripped)
            # Group consecutive imports into one block
            if code_type == 'import':
                start_line = i
                j = i + 1
                while j <= n:
                    next_line = lines[j-1].strip()
                    if next_line and not next_line.startswith('#') and self.classify_code_line(next_line) == 'import':
                        j += 1
                    else:
                        break
                # Create one import block representing the range [start_line, j-1]
                block_line_display = start_line if j == start_line + 1 else f"{start_line}-{j-1}"
                x = col_x['import']
                y = y_offset + col_count['import'] * 80
                col_count['import'] += 1
                block = CodeBlock(self.block_canvas, 'import', start_line, x, y)
                # Override text to show range
                display_text = f"Line {block_line_display}\nImport"
                self.block_canvas.itemconfig(block.text_id, text=display_text)
                self.code_blocks.append(block)
                tag = f"block_{id(block)}"
                self.block_tag_map[tag] = block
                # Click scrolls to start of group
                self.block_canvas.tag_bind(tag, "<Button-1>", lambda e, t=tag: self.handle_block_click(t))
                # Hover shows all grouped import lines
                self.block_canvas.tag_bind(tag, "<Enter>", lambda e, t=tag, s=start_line, e_line=j-1: self.handle_import_group_enter(t, e, s, e_line))
                self.block_canvas.tag_bind(tag, "<Leave>", lambda e, t=tag: self.handle_block_leave(t))
                i = j
                continue
            # Normal block for non-import types
            if code_type in categories_order:
                x = col_x[code_type]
                y = y_offset + col_count[code_type] * 80
                col_count[code_type] += 1
                block = CodeBlock(self.block_canvas, code_type, i, x, y)
                self.code_blocks.append(block)
                tag = f"block_{id(block)}"
                self.block_tag_map[tag] = block
                self.block_canvas.tag_bind(tag, "<Button-1>", lambda e, t=tag: self.handle_block_click(t))
                self.block_canvas.tag_bind(tag, "<Enter>", lambda e, t=tag: self.handle_block_enter(t, e))
                self.block_canvas.tag_bind(tag, "<Leave>", lambda e, t=tag: self.handle_block_leave(t))
            i += 1
        
        # Draw connections between blocks
        self.draw_block_connections()
    
    def classify_code_line(self, line):
        """Classify a line of code into a category"""
        line_lower = line.lower()
        
        if 'import' in line_lower:
            return 'import'
        elif any(keyword in line_lower for keyword in ['make_classification', 'make_regression', 'load_iris', 'fetch_california_housing']):
            return 'data_load'
        elif any(keyword in line_lower for keyword in ['train_test_split', 'standardscaler', 'labelencoder']):
            return 'preprocessing'
        elif any(keyword in line_lower for keyword in ['linearregression', 'logisticregression', 'decisiontree', 'randomforest']):
            return 'model'
        elif '.fit(' in line_lower:
            return 'training'
        elif '.predict(' in line_lower:
            return 'prediction'
        elif any(keyword in line_lower for keyword in ['plt.', 'plot', 'scatter', 'show()']):
            return 'visualization'
        elif any(keyword in line_lower for keyword in ['accuracy_score', 'mean_squared_error', 'classification_report']):
            return 'evaluation'
        else:
            return 'other'
    
    def draw_canvas_grid(self):
        """Draw grid on block canvas"""
        # Vertical lines
        for i in range(0, 1000, 50):
            self.block_canvas.create_line(i, 0, i, 2000, fill='#3c3c3c', width=1)
        
        # Horizontal lines
        for i in range(0, 2000, 50):
            self.block_canvas.create_line(0, i, 1000, i, fill='#3c3c3c', width=1)
    
    def draw_block_connections(self):
        """Draw connections between blocks with elbow lines"""
        for i in range(len(self.code_blocks) - 1):
            current_block = self.code_blocks[i]
            next_block = self.code_blocks[i + 1]
            sx = current_block.x + current_block.width // 2
            sy = current_block.y + current_block.height
            ex = next_block.x + next_block.width // 2
            ey = next_block.y
            mid_y = sy + 20
            points = [sx, sy, sx, mid_y, ex, mid_y, ex, ey]
            self.block_canvas.create_line(*points, fill='#4CAF50', width=2, arrow=tk.LAST)

    def handle_block_click(self, tag):
        """Handle block click: highlight block and related code line"""
        block = self.block_tag_map.get(tag)
        if not block:
            return
        for b in self.code_blocks:
            b.unhighlight()
        block.highlight()
        self.highlight_current_line(block.line_number)
        self.code_text.see(f"{block.line_number}.0")

    def handle_block_enter(self, tag, event):
        """Show tooltip with line content when hovering on block"""
        block = self.block_tag_map.get(tag)
        if not block:
            return
        line_text = self.code_text.get(f"{block.line_number}.0", f"{block.line_number}.end").strip()
        content = f"Line {block.line_number}: {line_text}" if line_text else f"Line {block.line_number}"
        if self.tooltip_window:
            try:
                self.tooltip_window.destroy()
            except Exception:
                pass
        tw = tk.Toplevel(self.root)
        tw.wm_overrideredirect(True)
        x = event.x_root + 10
        y = event.y_root + 10
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=content, bg='#333333', fg='white', font=('Consolas', 10), bd=1, relief=tk.SOLID)
        label.pack()
        self.tooltip_window = tw

    def handle_import_group_enter(self, tag, event, start_line, end_line):
        """Tooltip for grouped import lines shows the full import range content"""
        lines_content = []
        for ln in range(start_line, end_line + 1):
            text = self.code_text.get(f"{ln}.0", f"{ln}.end").strip()
            if text:
                lines_content.append(f"{ln}: {text}")
        content = "\n".join(lines_content) if lines_content else f"Lines {start_line}-{end_line}"
        if self.tooltip_window:
            try:
                self.tooltip_window.destroy()
            except Exception:
                pass
        tw = tk.Toplevel(self.root)
        tw.wm_overrideredirect(True)
        x = event.x_root + 10
        y = event.y_root + 10
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=content, bg='#333333', fg='white', font=('Consolas', 10), bd=1, relief=tk.SOLID, justify=tk.LEFT)
        label.pack()
        self.tooltip_window = tw

    def handle_block_leave(self, tag):
        """Hide tooltip when leaving block"""
        if self.tooltip_window:
            try:
                self.tooltip_window.destroy()
            except Exception:
                pass
            self.tooltip_window = None

    def on_canvas_zoom(self, event):
        """Zoom canvas with Ctrl+MouseWheel"""
        factor = 1.0
        if event.delta > 0:
            factor = 1.1
        elif event.delta < 0:
            factor = 0.9
        self.canvas_scale *= factor
        self.block_canvas.scale("all", event.x, event.y, factor, factor)
        bbox = self.block_canvas.bbox("all")
        if bbox:
            self.block_canvas.configure(scrollregion=bbox)

    def on_pan_start(self, event):
        """Start panning the canvas"""
        self.is_panning = True
        self.block_canvas.scan_mark(event.x, event.y)

    def on_pan_move(self, event):
        """Continue panning the canvas"""
        if self.is_panning:
            self.block_canvas.scan_dragto(event.x, event.y, gain=1)

    def on_pan_end(self, event):
        """End panning the canvas"""
        self.is_panning = False
    
    def custom_print(self, *args, **kwargs):
        """Custom print function to capture output"""
        output = ' '.join(str(arg) for arg in args)
        self.console_text.insert(tk.END, output + '\n', "output")
        self.console_text.see(tk.END)
    
    def show_templates(self):
        """Hiển thị cửa sổ Code Templates"""
        template_window = tk.Toplevel(self.root)
        template_window.title("ML Code Templates")
        template_window.geometry("800x600")
        template_window.configure(bg='#2b2b2b')
        
        # Title
        title_label = tk.Label(template_window, text="🚀 ML Algorithm Templates", 
                              bg='#2b2b2b', fg='#ffffff', 
                              font=('Consolas', 16, 'bold'))
        title_label.pack(pady=10)
        
        # Create notebook for categories
        notebook = ttk.Notebook(template_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Templates data
        templates = {
            "Classification": {
                "Linear Classification": '''# Linear Classification Example
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                          n_informative=2, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# Visualize
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis')
plt.title('Classification Results')
plt.show()''',
                
                "Decision Tree": '''# Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Accuracy: {accuracy:.2f}")'''
            },
            
            "Regression": {
                "Linear Regression": '''# Linear Regression Example
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualize
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Linear Regression Results')
plt.show()''',
                
                "Polynomial Regression": '''# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.linspace(0, 1, 100).reshape(-1, 1)
y = 2 * X.ravel() + 3 * X.ravel()**2 + np.random.normal(0, 0.1, 100)

# Create polynomial pipeline
poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])

# Train model
poly_model.fit(X, y)

# Predictions
y_pred = poly_model.predict(X)

# Visualize
plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X, y_pred, 'r-', label='Polynomial Fit')
plt.legend()
plt.title('Polynomial Regression')
plt.show()'''
            },
            
            "Data Processing": {
                "Data Loading & EDA": '''# Data Loading and Exploratory Data Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data (example with CSV)
# df = pd.read_csv('your_data.csv')

# For demo, create sample data
np.random.seed(42)
df = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 1000),
    'feature2': np.random.normal(2, 1.5, 1000),
    'target': np.random.randint(0, 2, 1000)
})

# Basic info
print("Dataset Info:")
print(df.info())
print("\\nDataset Description:")
print(df.describe())

# Check for missing values
print("\\nMissing Values:")
print(df.isnull().sum())

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Distribution plots
df['feature1'].hist(ax=axes[0,0], bins=30)
axes[0,0].set_title('Feature 1 Distribution')

df['feature2'].hist(ax=axes[0,1], bins=30)
axes[0,1].set_title('Feature 2 Distribution')

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, ax=axes[1,0])
axes[1,0].set_title('Correlation Matrix')

# Scatter plot
axes[1,1].scatter(df['feature1'], df['feature2'], c=df['target'], alpha=0.6)
axes[1,1].set_title('Feature Relationship')

plt.tight_layout()
plt.show()''',
                
                "Feature Engineering": '''# Feature Engineering Example
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif

# Sample data preparation
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# 1. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features scaled successfully")

# 2. Feature Selection
selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X_scaled, y)
selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
print(f"Selected features: {selected_features}")

# 3. Create new features (polynomial)
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_selected)
print(f"Original features: {X_selected.shape[1]}")
print(f"After polynomial features: {X_poly.shape[1]}")'''
            }
        }
        
        # Create tabs for each category
        for category, category_templates in templates.items():
            # Create frame for this category
            category_frame = tk.Frame(notebook, bg='#2b2b2b')
            notebook.add(category_frame, text=category)
            
            # Create scrollable text area
            text_frame = tk.Frame(category_frame, bg='#2b2b2b')
            text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Template list
            template_list = tk.Listbox(text_frame, bg='#404040', fg='white',
                                     font=('Consolas', 10), height=8)
            template_list.pack(fill=tk.X, pady=(0, 10))
            
            # Add templates to list
            for template_name in category_templates.keys():
                template_list.insert(tk.END, template_name)
            
            # Template code display
            code_display = tk.Text(text_frame, bg='#1e1e1e', fg='white',
                                 font=('Consolas', 9), wrap=tk.WORD)
            code_display.pack(fill=tk.BOTH, expand=True)
            
            # Scrollbar for code display
            scrollbar = tk.Scrollbar(code_display)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            code_display.config(yscrollcommand=scrollbar.set)
            scrollbar.config(command=code_display.yview)
            
            # Function to show selected template
            def show_template(event, templates_dict=category_templates, display=code_display):
                selection = event.widget.curselection()
                if selection:
                    template_name = event.widget.get(selection[0])
                    template_code = templates_dict[template_name]
                    display.delete(1.0, tk.END)
                    display.insert(1.0, template_code)
            
            template_list.bind('<<ListboxSelect>>', show_template)
            
            # Insert button
            def insert_template(templates_dict=category_templates, display=code_display):
                template_code = display.get(1.0, tk.END)
                if template_code.strip():
                    self.code_text.insert(tk.END, f"\\n{template_code}\\n")
                    self.code_text.see(tk.END)
                    template_window.destroy()
            
            insert_btn = tk.Button(text_frame, text="📝 Insert Template",
                                 command=insert_template,
                                 bg='#4CAF50', fg='white',
                                 font=('Consolas', 10),
                                 relief=tk.FLAT, padx=20)
            insert_btn.pack(pady=10)

    def init_3d_view(self):
        """3D visualizer temporarily disabled"""
        self.three_d_initialized = False

    def switch_to_3d_view(self):
        """Temporarily disabled 3D visualizer"""
        try:
            messagebox.showinfo("3D Visualizer", "Tính năng 3D tạm thời vô hiệu hóa. Sẽ phát triển sau.")
        except Exception:
            pass
        # Keep Blocks view as default
        if hasattr(self, 'block_view_frame'):
            self.block_view_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        if hasattr(self, 'visual_title'):
            self.visual_title.config(text="🧩 Live Block Visualization")

    def switch_to_blocks_view(self):
        """Ensure Blocks view is shown; 3D hidden"""
        if hasattr(self, 'three_d_view_frame'):
            self.three_d_view_frame.pack_forget()
        if hasattr(self, 'visual_title'):
            self.visual_title.config(text="🧩 Live Block Visualization")
        if hasattr(self, 'block_view_frame'):
            self.block_view_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def show_3d_visualizer(self):
        """Temporarily disabled 3D visualizer; keep Blocks view"""
        self.switch_to_3d_view()

    # Removed toggle_3d_builder per request
        
    def add_3d_layer(self, layer_name, layer_info):
        """Disabled while 3D is off"""
        return
        
    def draw_3d_model(self):
        """Disabled while 3D is off"""
        return
    
    def darken_color(self, color):
        """Làm tối màu để tạo hiệu ứng 3D"""
        # Convert hex to RGB
        color = color.lstrip('#')
        rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        # Darken by 30%
        darkened = tuple(int(c * 0.7) for c in rgb)
        # Convert back to hex
        return f"#{darkened[0]:02x}{darkened[1]:02x}{darkened[2]:02x}"
    
    def draw_3d_grid(self):
        """Disabled while 3D is off"""
        return
    
    def rotate_3d_model(self):
        """Disabled while 3D is off"""
        return
    
    def zoom_in_3d(self):
        """Disabled while 3D is off"""
        return
    
    def zoom_out_3d(self):
        """Disabled while 3D is off"""
        return
    
    def clear_3d_model(self):
        """Disabled while 3D is off"""
        return
    
    def load_3d_template(self, template_name):
        """Load template mô hình 3D có sẵn"""
        self.clear_3d_model()
        
        templates = {
            "Simple Neural Network": [
                {"name": "Input Layer", "color": "#4CAF50", "icon": "📥"},
                {"name": "Dense/Linear", "color": "#2196F3", "icon": "🔗"},
                {"name": "Activation", "color": "#FFEB3B", "icon": "⚡"},
                {"name": "Dense/Linear", "color": "#2196F3", "icon": "🔗"},
                {"name": "Output Layer", "color": "#795548", "icon": "📤"}
            ],
            "CNN for Image Classification": [
                {"name": "Input Layer", "color": "#4CAF50", "icon": "📥"},
                {"name": "Convolutional", "color": "#FF9800", "icon": "🔲"},
                {"name": "Activation", "color": "#FFEB3B", "icon": "⚡"},
                {"name": "Convolutional", "color": "#FF9800", "icon": "🔲"},
                {"name": "BatchNorm", "color": "#00BCD4", "icon": "📊"},
                {"name": "Dense/Linear", "color": "#2196F3", "icon": "🔗"},
                {"name": "Dropout", "color": "#F44336", "icon": "❌"},
                {"name": "Output Layer", "color": "#795548", "icon": "📤"}
            ],
            "LSTM for Time Series": [
                {"name": "Input Layer", "color": "#4CAF50", "icon": "📥"},
                {"name": "LSTM/RNN", "color": "#9C27B0", "icon": "🔄"},
                {"name": "LSTM/RNN", "color": "#9C27B0", "icon": "🔄"},
                {"name": "Dropout", "color": "#F44336", "icon": "❌"},
                {"name": "Dense/Linear", "color": "#2196F3", "icon": "🔗"},
                {"name": "Output Layer", "color": "#795548", "icon": "📤"}
            ],
            "Keras Sequential Model": [
                {"name": "Input Layer", "color": "#4CAF50", "icon": "📥"},
                {"name": "Dense/Linear", "color": "#2196F3", "icon": "🔗"},
                {"name": "Activation", "color": "#FFEB3B", "icon": "⚡"},
                {"name": "Dropout", "color": "#F44336", "icon": "❌"},
                {"name": "Dense/Linear", "color": "#2196F3", "icon": "🔗"},
                {"name": "Output Layer", "color": "#795548", "icon": "📤"}
            ],
            "Transformer Architecture": [
                {"name": "Input Layer", "color": "#4CAF50", "icon": "📥"},
                {"name": "Dense/Linear", "color": "#2196F3", "icon": "🔗"},
                {"name": "Dense/Linear", "color": "#2196F3", "icon": "🔗"},
                {"name": "Dense/Linear", "color": "#2196F3", "icon": "🔗"},
                {"name": "Dropout", "color": "#F44336", "icon": "❌"},
                {"name": "Dense/Linear", "color": "#2196F3", "icon": "🔗"},
                {"name": "Output Layer", "color": "#795548", "icon": "📤"}
            ],
            "ResNet Block": [
                {"name": "Input Layer", "color": "#4CAF50", "icon": "📥"},
                {"name": "Convolutional", "color": "#FF9800", "icon": "🔲"},
                {"name": "BatchNorm", "color": "#00BCD4", "icon": "📊"},
                {"name": "Activation", "color": "#FFEB3B", "icon": "⚡"},
                {"name": "Convolutional", "color": "#FF9800", "icon": "🔲"},
                {"name": "BatchNorm", "color": "#00BCD4", "icon": "📊"},
                {"name": "Output Layer", "color": "#795548", "icon": "📤"}
            ],
            "U-Net for Segmentation": [
                {"name": "Input Layer", "color": "#4CAF50", "icon": "📥"},
                {"name": "Convolutional", "color": "#FF9800", "icon": "🔲"},
                {"name": "Convolutional", "color": "#FF9800", "icon": "🔲"},
                {"name": "Convolutional", "color": "#FF9800", "icon": "🔲"},
                {"name": "Convolutional", "color": "#FF9800", "icon": "🔲"},
                {"name": "Convolutional", "color": "#FF9800", "icon": "🔲"},
                {"name": "Output Layer", "color": "#795548", "icon": "📤"}
            ]
        }
        
        if template_name in templates:
            for layer_info in templates[template_name]:
                layer_data = {
                    'name': layer_info['name'],
                    'color': layer_info['color'],
                    'icon': layer_info['icon'],
                    'position': len(self.model_layers_3d),
                    'width': 200,
                    'height': 60,
                    'depth': 40
                }
                self.model_layers_3d.append(layer_data)
            
            self.draw_3d_model()
    
    def export_3d_model_code(self):
        """Xuất code Python từ mô hình 3D"""
        if not self.model_layers_3d:
            messagebox.showwarning("Warning", "Chưa có layer nào trong mô hình!")
            return
        
        # Generate code based on layers
        code_lines = [
            "# Generated Model Architecture",
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
            "",
            "class GeneratedModel(nn.Module):",
            "    def __init__(self):",
            "        super(GeneratedModel, self).__init__()"
        ]
        
        # Add layers
        layer_count = {}
        for layer in self.model_layers_3d:
            layer_type = layer['name']
            if layer_type not in layer_count:
                layer_count[layer_type] = 0
            layer_count[layer_type] += 1
            
            if layer_type == "Dense/Linear":
                code_lines.append(f"        self.linear{layer_count[layer_type]} = nn.Linear(128, 64)")
            elif layer_type == "Convolutional":
                code_lines.append(f"        self.conv{layer_count[layer_type]} = nn.Conv2d(3, 32, 3)")
            elif layer_type == "LSTM/RNN":
                code_lines.append(f"        self.lstm{layer_count[layer_type]} = nn.LSTM(128, 64)")
            elif layer_type == "Dropout":
                code_lines.append(f"        self.dropout{layer_count[layer_type]} = nn.Dropout(0.5)")
            elif layer_type == "BatchNorm":
                code_lines.append(f"        self.batchnorm{layer_count[layer_type]} = nn.BatchNorm1d(64)")
        
        code_lines.extend([
            "",
            "    def forward(self, x):",
            "        # Add your forward pass logic here",
            "        return x"
        ])
        
        # Insert code into editor
        generated_code = "\n".join(code_lines)
        current_code = self.code_text.get("1.0", tk.END)
        if current_code.strip():
            generated_code = "\n\n" + generated_code
        
        self.code_text.insert(tk.END, generated_code)
        messagebox.showinfo("Success", "Code đã được xuất vào editor!")

    def run_code(self):
        """Run the entire code"""
        if self.is_running:
            return
        
        self.is_running = True
        self.console_text.delete(1.0, tk.END)
        self.console_text.insert(tk.END, "🚀 Running code...\n", "info")
        
        threading.Thread(target=self.execute_code, daemon=True).start()
    
    def run_step_by_step(self):
        """Run code step by step with visualization"""
        if self.is_running:
            return
        
        self.is_running = True
        self.console_text.delete(1.0, tk.END)
        self.console_text.insert(tk.END, "🔍 Running step by step...\n", "info")
        
        threading.Thread(target=self.execute_step_by_step, daemon=True).start()
    
    def execute_code(self):
        """Execute the code"""
        try:
            code = self.code_text.get(1.0, tk.END)
            
            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            # Execute code
            exec(code, self.execution_globals)
            
            # Get output
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            
            if output:
                self.root.after(0, lambda: self.console_text.insert(tk.END, output, "output"))
            
            # Update variables display
            self.root.after(0, self.update_variables_display)
            
            # Update visualization if matplotlib was used
            self.root.after(0, self.update_plot_display)
            
        except Exception as e:
            self.root.after(0, lambda: self.console_text.insert(tk.END, f"❌ Error: {str(e)}\n", "error"))
        finally:
            self.is_running = False
    
    def execute_step_by_step(self):
        """Execute code step by step with block highlighting"""
        try:
            code = self.code_text.get(1.0, tk.END)
            lines = code.split('\n')
            
            for i, line in enumerate(lines, 1):
                if not self.is_running:
                    break
                
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Highlight current line in code editor
                self.root.after(0, lambda l=i: self.highlight_current_line(l))
                
                # Highlight corresponding block
                matching_blocks = [b for b in self.code_blocks if b.line_number == i]
                if matching_blocks:
                    block = matching_blocks[0]
                    self.root.after(0, lambda b=block: b.highlight())
                
                # Execute line
                try:
                    exec(line, self.execution_globals)
                    self.root.after(0, lambda l=line: self.console_text.insert(tk.END, f"✅ Executed: {l}\n", "info"))
                except Exception as e:
                    self.root.after(0, lambda e=e: self.console_text.insert(tk.END, f"❌ Error: {str(e)}\n", "error"))
                
                # Update displays
                self.root.after(0, self.update_variables_display)
                self.root.after(0, self.update_plot_display)
                
                # Wait based on speed setting
                time.sleep(1.0 / self.speed_var.get())
                
                # Unhighlight block
                if matching_blocks:
                    self.root.after(0, lambda b=matching_blocks[0]: b.unhighlight())
            
            # Remove line highlighting
            self.root.after(0, self.remove_line_highlighting)
            
        except Exception as e:
            self.root.after(0, lambda: self.console_text.insert(tk.END, f"❌ Error: {str(e)}\n", "error"))
        finally:
            self.is_running = False
    
    def highlight_current_line(self, line_number):
        """Highlight current line in code editor"""
        self.code_text.tag_remove("current_line", 1.0, tk.END)
        start_idx = f"{line_number}.0"
        end_idx = f"{line_number}.end"
        self.code_text.tag_add("current_line", start_idx, end_idx)
    
    def remove_line_highlighting(self):
        """Remove line highlighting"""
        self.code_text.tag_remove("current_line", 1.0, tk.END)
    
    def update_variables_display(self):
        """Update the variables display"""
        self.vars_text.delete(1.0, tk.END)
        
        vars_info = "=== VARIABLES ===\n\n"
        
        for name, value in self.execution_globals.items():
            if not name.startswith('_') and name not in ['np', 'pd', 'plt', 'print']:
                try:
                    if hasattr(value, 'shape'):
                        vars_info += f"{name}: {type(value).__name__} {value.shape}\n"
                    elif isinstance(value, (int, float, str, bool)):
                        vars_info += f"{name}: {value}\n"
                    elif hasattr(value, '__class__'):
                        vars_info += f"{name}: {type(value).__name__}\n"
                except:
                    vars_info += f"{name}: {type(value).__name__}\n"
        
        self.vars_text.insert(1.0, vars_info)
    
    def update_plot_display(self):
        """Update the plot display"""
        # Check if there are any plots to display
        if 'X' in self.execution_globals and 'y' in self.execution_globals:
            self.ax.clear()
            self.ax.set_facecolor('#2b2b2b')
            
            X = self.execution_globals['X']
            y = self.execution_globals['y']
            
            if len(X.shape) == 2 and X.shape[1] >= 2:
                scatter = self.ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
                self.ax.set_xlabel('Feature 1', color='white')
                self.ax.set_ylabel('Feature 2', color='white')
                self.ax.set_title('Data Visualization', color='white')
            
            self.ax.tick_params(colors='white')
            self.fig.tight_layout()
            self.plot_canvas.draw()
    
    def stop_execution(self):
        """Stop code execution"""
        self.is_running = False
        self.remove_line_highlighting()
        for block in self.code_blocks:
            block.unhighlight()
        self.console_text.insert(tk.END, "⏹ Execution stopped\n", "info")
    
    def reset_environment(self):
        """Reset the execution environment"""
        self.execution_globals.clear()
        self.setup_execution_environment()
        self.console_text.delete(1.0, tk.END)
        self.vars_text.delete(1.0, tk.END)
        self.ax.clear()
        self.plot_canvas.draw()
        self.console_text.insert(tk.END, "🔄 Environment reset\n", "info")
    
    def save_code(self):
        """Save code to file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".py",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")]
        )
        
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.code_text.get(1.0, tk.END))
            self.console_text.insert(tk.END, f"💾 Code saved to {filename}\n", "info")
    
    def load_code(self):
        """Load code from file"""
        filename = filedialog.askopenfilename(
            filetypes=[("Python files", "*.py"), ("All files", "*.*")]
        )
        
        if filename:
            with open(filename, 'r', encoding='utf-8') as f:
                code = f.read()
            
            self.code_text.delete(1.0, tk.END)
            self.code_text.insert(1.0, code)
            self.on_code_change()
            self.console_text.insert(tk.END, f"📁 Code loaded from {filename}\n", "info")
    
    def load_sample_code(self):
        """Load sample ML code"""
        sample_code = '''# Sample Machine Learning Code
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generate sample data
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                          n_informative=2, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
'''
        
        self.code_text.insert(1.0, sample_code)
        self.on_code_change()
    
    def load_help_content(self):
        """Load help content"""
        help_content = '''🐍 ML Code Editor Help

📝 FEATURES:
• Live syntax highlighting
• Block visualization from code
• Step-by-step execution
• Variable monitoring
• Plot visualization
• 🧠 Layers: CNN animation Input→Conv→ReLU→Pool

⌨️ SHORTCUTS:
• Ctrl+S: Save code
• Ctrl+O: Load code
• F5: Run code
• F10: Step by step
• F7: Run Layers animation
• F9: Toggle Dev Focus
• Ctrl+Enter: Run selection/current line
• Shift+F5: Stop execution

🧩 SUPPORTED LIBRARIES:
• NumPy (np)
• Pandas (pd)
• Matplotlib (plt)
• Scikit-learn (sklearn)

📊 BLOCK TYPES:
• Import: Library imports
• Data Load: Dataset loading
• Preprocessing: Data preparation
• Model: ML algorithms
• Training: Model fitting
• Prediction: Making predictions
• Visualization: Plotting
• Evaluation: Performance metrics

💡 TIPS:
• Write clean, readable code
• Use meaningful variable names
• Add comments for clarity
• Test step by step for debugging

🖱 BLOCK INTERACTION:
• Click block: highlight và cuộn tới dòng code tương ứng
• Hover block: hiển thị tooltip nội dung dòng code
• Ctrl + Mouse Wheel: zoom canvas blocks
• Middle mouse drag hoặc Shift + Drag: pan canvas

🧠 LAYERS ANIMATION:
• Mở tab "🧠 Layers" trên panel phải
• Vẽ digit 28×28 ở ô bên trái
• Nhấn ▶ Predict hoặc phím F7 để chạy
• Quan sát 4 ô: Input → Conv → ReLU → Pool
• Điều chỉnh tốc độ bằng thanh "Speed (ms)"
• Nhấn 🧹 Clear để xoá và vẽ lại
'''
        
        self.help_text.insert(1.0, help_content)

def main():
    root = tk.Tk()
    app = MLCodeEditor(root)
    root.mainloop()

if __name__ == "__main__":
    main()