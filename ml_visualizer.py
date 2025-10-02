import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import time
import json
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd

class MLBlock:
    """Represents a draggable ML algorithm block"""
    def __init__(self, canvas, block_type, x, y, width=120, height=60):
        self.canvas = canvas
        self.block_type = block_type
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.is_selected = False
        self.is_running = False
        self.connections = []
        
        # Block colors based on type
        self.colors = {
            'data': '#4CAF50',      # Green for data
            'preprocessing': '#2196F3',  # Blue for preprocessing
            'model': '#FF9800',     # Orange for models
            'evaluation': '#9C27B0', # Purple for evaluation
            'visualization': '#F44336'  # Red for visualization
        }
        
        self.create_block()
        
    def create_block(self):
        """Create the visual block on canvas"""
        color = self.colors.get(self.get_block_category(), '#757575')
        
        # Main rectangle
        self.rect_id = self.canvas.create_rectangle(
            self.x, self.y, self.x + self.width, self.y + self.height,
            fill=color, outline='white', width=2, tags=f"block_{id(self)}"
        )
        
        # Block title
        self.text_id = self.canvas.create_text(
            self.x + self.width//2, self.y + self.height//2,
            text=self.block_type, fill='white', font=('Arial', 10, 'bold'),
            tags=f"block_{id(self)}"
        )
        
        # Input/Output ports
        self.create_ports()
        
        # Bind events
        self.canvas.tag_bind(f"block_{id(self)}", "<Button-1>", self.on_click)
        
    def create_ports(self):
        """Create input/output connection ports"""
        port_size = 8
        
        # Input port (left side)
        if self.block_type not in ['Load Data', 'Generate Data']:
            self.input_port = self.canvas.create_oval(
                self.x - port_size//2, self.y + self.height//2 - port_size//2,
                self.x + port_size//2, self.y + self.height//2 + port_size//2,
                fill='lightblue', outline='blue', width=2,
                tags=f"block_{id(self)}"
            )
        
        # Output port (right side)
        self.output_port = self.canvas.create_oval(
            self.x + self.width - port_size//2, self.y + self.height//2 - port_size//2,
            self.x + self.width + port_size//2, self.y + self.height//2 + port_size//2,
            fill='lightgreen', outline='green', width=2,
            tags=f"block_{id(self)}"
        )
    
    def get_block_category(self):
        """Get the category of the block"""
        data_blocks = ['Load Data', 'Generate Data']
        preprocessing_blocks = ['Train/Test Split', 'Normalize', 'Feature Selection']
        model_blocks = ['Linear Regression', 'Logistic Regression', 'Decision Tree', 'Random Forest']
        evaluation_blocks = ['Accuracy', 'MSE', 'Confusion Matrix']
        visualization_blocks = ['Scatter Plot', 'Decision Boundary', 'Loss Curve']
        
        if self.block_type in data_blocks:
            return 'data'
        elif self.block_type in preprocessing_blocks:
            return 'preprocessing'
        elif self.block_type in model_blocks:
            return 'model'
        elif self.block_type in evaluation_blocks:
            return 'evaluation'
        elif self.block_type in visualization_blocks:
            return 'visualization'
        return 'other'
    
    def on_click(self, event):
        """Handle block click"""
        self.canvas.tag_raise(f"block_{id(self)}")
        
    def on_drag(self, event):
        """Handle block dragging"""
        if self.is_selected:
            dx = event.x - (self.x + self.width//2)
            dy = event.y - (self.y + self.height//2)
            self.move(dx, dy)
    
    def on_release(self, event):
        """Handle block release"""
        self.is_selected = False
    
    def move(self, dx, dy):
        """Move the block by dx, dy"""
        self.x += dx
        self.y += dy
        self.canvas.move(f"block_{id(self)}", dx, dy)
    
    def highlight(self, color='#FFD700'):
        """Highlight the block during execution"""
        self.canvas.itemconfig(self.rect_id, outline=color, width=4)
        self.is_running = True
    
    def unhighlight(self):
        """Remove highlight"""
        self.canvas.itemconfig(self.rect_id, outline='white', width=2)
        self.is_running = False
    
    def execute(self, input_data=None):
        """Execute the block's functionality"""
        if self.block_type == 'Generate Data':
            return self.generate_data()
        elif self.block_type == 'Linear Regression':
            return self.linear_regression(input_data)
        elif self.block_type == 'Decision Tree':
            return self.decision_tree(input_data)
        elif self.block_type == 'Train/Test Split':
            return self.train_test_split(input_data)
        # Add more block implementations
        return input_data
    
    def generate_data(self):
        """Generate sample data"""
        X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                                 n_informative=2, random_state=42, n_clusters_per_class=1)
        return {'X': X, 'y': y, 'type': 'classification'}
    
    def linear_regression(self, data):
        """Perform linear regression"""
        if data and 'X_train' in data:
            model = LinearRegression()
            model.fit(data['X_train'], data['y_train'])
            y_pred = model.predict(data['X_test'])
            return {**data, 'model': model, 'y_pred': y_pred}
        return data
    
    def decision_tree(self, data):
        """Perform decision tree classification"""
        if data and 'X_train' in data:
            model = DecisionTreeClassifier(random_state=42)
            model.fit(data['X_train'], data['y_train'])
            y_pred = model.predict(data['X_test'])
            return {**data, 'model': model, 'y_pred': y_pred}
        return data
    
    def train_test_split(self, data):
        """Split data into train/test sets"""
        if data and 'X' in data:
            X_train, X_test, y_train, y_test = train_test_split(
                data['X'], data['y'], test_size=0.3, random_state=42
            )
            return {
                **data,
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train, 'y_test': y_test
            }
        return data

class MLVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 ML Algorithm Visualizer - Block Programming")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#2b2b2b')
        
        # State variables
        self.blocks = []
        self.connections = []
        self.is_running = False
        self.current_data = None
        self.execution_order = []
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the main UI"""
        # Main container
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top toolbar
        self.create_toolbar(main_frame)
        
        # Main content area
        content_frame = tk.Frame(main_frame, bg='#2b2b2b')
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Left panel - Block palette
        self.create_block_palette(content_frame)
        
        # Center panel - Canvas for blocks
        self.create_canvas_area(content_frame)
        
        # Right panel - Visualization and code
        self.create_visualization_panel(content_frame)
        
        # Bottom panel - Logs and status
        self.create_status_panel(main_frame)
        
    def create_toolbar(self, parent):
        """Create the top toolbar"""
        toolbar = tk.Frame(parent, bg='#3c3c3c', relief=tk.RAISED, bd=2)
        toolbar.pack(fill=tk.X, pady=(0, 5))
        
        # Control buttons
        tk.Button(toolbar, text="▶ Run Pipeline", command=self.run_pipeline,
                 bg='#4CAF50', fg='white', font=('Arial', 12, 'bold')).pack(side=tk.LEFT, padx=5, pady=5)
        
        tk.Button(toolbar, text="⏹ Stop", command=self.stop_execution,
                 bg='#F44336', fg='white', font=('Arial', 12, 'bold')).pack(side=tk.LEFT, padx=5, pady=5)
        
        tk.Button(toolbar, text="🔄 Reset", command=self.reset_canvas,
                 bg='#FF9800', fg='white', font=('Arial', 12, 'bold')).pack(side=tk.LEFT, padx=5, pady=5)
        
        tk.Button(toolbar, text="💾 Save", command=self.save_pipeline,
                 bg='#2196F3', fg='white', font=('Arial', 12, 'bold')).pack(side=tk.LEFT, padx=5, pady=5)
        
        tk.Button(toolbar, text="📁 Load", command=self.load_pipeline,
                 bg='#9C27B0', fg='white', font=('Arial', 12, 'bold')).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Speed control
        tk.Label(toolbar, text="Animation Speed:", bg='#3c3c3c', fg='white', 
                font=('Arial', 10)).pack(side=tk.RIGHT, padx=(20, 5), pady=5)
        
        self.speed_var = tk.DoubleVar(value=1.0)
        self.speed_scale = tk.Scale(toolbar, from_=0.1, to=3.0, resolution=0.1, 
                                   orient=tk.HORIZONTAL, variable=self.speed_var,
                                   bg='#3c3c3c', fg='white', font=('Arial', 9))
        self.speed_scale.pack(side=tk.RIGHT, padx=5, pady=5)
        
    def create_block_palette(self, parent):
        """Create the left panel with draggable blocks"""
        palette_frame = tk.Frame(parent, bg='#1e1e1e', relief=tk.RAISED, bd=2, width=200)
        palette_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        palette_frame.pack_propagate(False)
        
        tk.Label(palette_frame, text="🧩 ML Blocks", bg='#1e1e1e', fg='#4CAF50',
                font=('Arial', 14, 'bold')).pack(pady=10)
        
        # Block categories
        categories = {
            "📊 Data": ['Generate Data', 'Load Data'],
            "🔧 Preprocessing": ['Train/Test Split', 'Normalize', 'Feature Selection'],
            "🤖 Models": ['Linear Regression', 'Logistic Regression', 'Decision Tree', 'Random Forest'],
            "📈 Evaluation": ['Accuracy', 'MSE', 'Confusion Matrix'],
            "📉 Visualization": ['Scatter Plot', 'Decision Boundary', 'Loss Curve']
        }
        
        for category, blocks in categories.items():
            # Category header
            category_frame = tk.Frame(palette_frame, bg='#2c2c2c', relief=tk.RAISED, bd=1)
            category_frame.pack(fill=tk.X, padx=5, pady=2)
            
            tk.Label(category_frame, text=category, bg='#2c2c2c', fg='white',
                    font=('Arial', 11, 'bold')).pack(pady=3)
            
            # Blocks in category
            for block_name in blocks:
                block_btn = tk.Button(palette_frame, text=block_name,
                                    command=lambda b=block_name: self.add_block_to_canvas(b),
                                    bg='#4a4a4a', fg='white', font=('Arial', 9),
                                    relief=tk.FLAT, bd=1)
                block_btn.pack(fill=tk.X, padx=10, pady=1)
                
                # Hover effects
                block_btn.bind("<Enter>", lambda e, btn=block_btn: btn.config(bg='#5a5a5a'))
                block_btn.bind("<Leave>", lambda e, btn=block_btn: btn.config(bg='#4a4a4a'))
    
    def create_canvas_area(self, parent):
        """Create the center canvas area for blocks"""
        canvas_frame = tk.Frame(parent, bg='#1e1e1e', relief=tk.RAISED, bd=2)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        tk.Label(canvas_frame, text="🎨 ML Pipeline Canvas", bg='#1e1e1e', fg='#4CAF50',
                font=('Arial', 14, 'bold')).pack(pady=5)
        
        # Canvas with scrollbars
        canvas_container = tk.Frame(canvas_frame, bg='#1e1e1e')
        canvas_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(canvas_container, bg='#2b2b2b', 
                               scrollregion=(0, 0, 2000, 2000))
        
        # Scrollbars
        v_scrollbar = tk.Scrollbar(canvas_container, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = tk.Scrollbar(canvas_container, orient=tk.HORIZONTAL, command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack scrollbars and canvas
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Grid lines for better visual guidance
        self.draw_grid()
        
    def create_visualization_panel(self, parent):
        """Create the right panel for visualization and code"""
        viz_frame = tk.Frame(parent, bg='#1e1e1e', relief=tk.RAISED, bd=2, width=400)
        viz_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        viz_frame.pack_propagate(False)
        
        # Notebook for tabs
        notebook = ttk.Notebook(viz_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Visualization tab
        viz_tab = tk.Frame(notebook, bg='#1e1e1e')
        notebook.add(viz_tab, text="📊 Visualization")
        
        # Matplotlib figure
        self.fig = Figure(figsize=(5, 4), dpi=100, facecolor='#2b2b2b')
        self.ax = self.fig.add_subplot(111, facecolor='#2b2b2b')
        self.ax.tick_params(colors='white')
        
        self.canvas_plot = FigureCanvasTkAgg(self.fig, viz_tab)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Code tab
        code_tab = tk.Frame(notebook, bg='#1e1e1e')
        notebook.add(code_tab, text="💻 Generated Code")
        
        self.code_text = tk.Text(code_tab, bg='#1e1e1e', fg='white', 
                                font=('Consolas', 10), wrap=tk.WORD)
        code_scrollbar = tk.Scrollbar(code_tab, command=self.code_text.yview)
        self.code_text.configure(yscrollcommand=code_scrollbar.set)
        
        code_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.code_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Data tab
        data_tab = tk.Frame(notebook, bg='#1e1e1e')
        notebook.add(data_tab, text="📋 Data Info")
        
        self.data_text = tk.Text(data_tab, bg='#1e1e1e', fg='white',
                                font=('Consolas', 9), wrap=tk.WORD)
        data_scrollbar = tk.Scrollbar(data_tab, command=self.data_text.yview)
        self.data_text.configure(yscrollcommand=data_scrollbar.set)
        
        data_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.data_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
    def create_status_panel(self, parent):
        """Create the bottom status panel"""
        status_frame = tk.Frame(parent, bg='#1e1e1e', relief=tk.RAISED, bd=2, height=150)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        status_frame.pack_propagate(False)
        
        tk.Label(status_frame, text="📋 Execution Log", bg='#1e1e1e', fg='#4CAF50',
                font=('Arial', 12, 'bold')).pack(pady=5)
        
        # Log text area
        log_container = tk.Frame(status_frame, bg='#1e1e1e')
        log_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_text = tk.Text(log_container, bg='#1e1e1e', fg='white',
                               font=('Consolas', 9), height=8)
        log_scrollbar = tk.Scrollbar(log_container, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure log tags
        self.log_text.tag_configure("info", foreground="#4CAF50")
        self.log_text.tag_configure("warning", foreground="#FF9800")
        self.log_text.tag_configure("error", foreground="#F44336")
        self.log_text.tag_configure("success", foreground="#8BC34A")
        
    def draw_grid(self):
        """Draw grid lines on canvas"""
        # Vertical lines
        for i in range(0, 2000, 50):
            self.canvas.create_line(i, 0, i, 2000, fill='#3c3c3c', width=1)
        
        # Horizontal lines
        for i in range(0, 2000, 50):
            self.canvas.create_line(0, i, 2000, i, fill='#3c3c3c', width=1)
    
    def add_block_to_canvas(self, block_type):
        """Add a new block to the canvas"""
        # Find a good position (avoid overlapping)
        x = 100 + (len(self.blocks) % 5) * 150
        y = 100 + (len(self.blocks) // 5) * 100
        
        block = MLBlock(self.canvas, block_type, x, y)
        self.blocks.append(block)
        
        self.log_message(f"✅ Added {block_type} block to canvas", "info")
        self.update_generated_code()
    
    def run_pipeline(self):
        """Run the ML pipeline with animation"""
        if not self.blocks:
            self.log_message("⚠️ No blocks to execute!", "warning")
            return
        
        if self.is_running:
            self.log_message("⚠️ Pipeline is already running!", "warning")
            return
        
        self.is_running = True
        self.log_message("🚀 Starting ML pipeline execution...", "info")
        
        # Run in separate thread to avoid blocking UI
        threading.Thread(target=self.execute_pipeline, daemon=True).start()
    
    def execute_pipeline(self):
        """Execute the pipeline with animations"""
        try:
            # Determine execution order (simple left-to-right for now)
            sorted_blocks = sorted(self.blocks, key=lambda b: b.x)
            
            current_data = None
            
            for i, block in enumerate(sorted_blocks):
                if not self.is_running:
                    break
                
                # Highlight current block
                self.root.after(0, lambda b=block: b.highlight())
                self.root.after(0, lambda b=block: self.log_message(f"🔄 Executing {b.block_type}...", "info"))
                
                # Animate data flow (dots moving between blocks)
                if i > 0:
                    self.root.after(0, lambda prev=sorted_blocks[i-1], curr=block: 
                                  self.animate_data_flow(prev, curr))
                
                # Execute block
                time.sleep(1.0 / self.speed_var.get())  # Animation delay
                current_data = block.execute(current_data)
                
                # Update visualization
                if current_data:
                    self.root.after(0, lambda data=current_data: self.update_visualization(data))
                    self.root.after(0, lambda data=current_data: self.update_data_info(data))
                
                # Unhighlight block
                self.root.after(0, lambda b=block: b.unhighlight())
                self.root.after(0, lambda b=block: self.log_message(f"✅ {b.block_type} completed", "success"))
                
                time.sleep(0.5 / self.speed_var.get())  # Pause between blocks
            
            self.current_data = current_data
            self.root.after(0, lambda: self.log_message("🎉 Pipeline execution completed!", "success"))
            
        except Exception as e:
            self.root.after(0, lambda: self.log_message(f"❌ Error: {str(e)}", "error"))
        finally:
            self.is_running = False
    
    def animate_data_flow(self, from_block, to_block):
        """Animate data flowing between blocks"""
        start_x = from_block.x + from_block.width
        start_y = from_block.y + from_block.height // 2
        end_x = to_block.x
        end_y = to_block.y + to_block.height // 2
        
        # Create animated dots
        dots = []
        for i in range(5):
            dot = self.canvas.create_oval(start_x-3, start_y-3, start_x+3, start_y+3,
                                        fill='#FFD700', outline='#FFA000')
            dots.append(dot)
        
        # Animate dots movement
        steps = 20
        for step in range(steps):
            for i, dot in enumerate(dots):
                if step >= i * 2:  # Stagger the dots
                    progress = (step - i * 2) / (steps - i * 2) if step >= i * 2 else 0
                    x = start_x + (end_x - start_x) * progress
                    y = start_y + (end_y - start_y) * progress
                    self.canvas.coords(dot, x-3, y-3, x+3, y+3)
            
            self.canvas.update()
            time.sleep(0.05 / self.speed_var.get())
        
        # Remove dots
        for dot in dots:
            self.canvas.delete(dot)
    
    def update_visualization(self, data):
        """Update the visualization panel"""
        self.ax.clear()
        self.ax.set_facecolor('#2b2b2b')
        
        if 'X' in data and 'y' in data:
            X, y = data['X'], data['y']
            
            # Scatter plot of data
            scatter = self.ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
            self.ax.set_xlabel('Feature 1', color='white')
            self.ax.set_ylabel('Feature 2', color='white')
            self.ax.set_title('Data Visualization', color='white')
            
            # Add decision boundary if model exists
            if 'model' in data and hasattr(data['model'], 'predict'):
                self.plot_decision_boundary(data['model'], X, y)
        
        self.ax.tick_params(colors='white')
        self.fig.tight_layout()
        self.canvas_plot.draw()
    
    def plot_decision_boundary(self, model, X, y):
        """Plot decision boundary for classification models"""
        try:
            h = 0.02
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                               np.arange(y_min, y_max, h))
            
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            self.ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
        except:
            pass  # Skip if model doesn't support decision boundary
    
    def update_data_info(self, data):
        """Update data information panel"""
        self.data_text.delete(1.0, tk.END)
        
        info = "=== DATA INFORMATION ===\n\n"
        
        if 'X' in data:
            info += f"📊 Features shape: {data['X'].shape}\n"
            info += f"📈 Target shape: {data['y'].shape}\n\n"
        
        if 'X_train' in data:
            info += f"🏋️ Training set: {data['X_train'].shape}\n"
            info += f"🧪 Test set: {data['X_test'].shape}\n\n"
        
        if 'model' in data:
            info += f"🤖 Model: {type(data['model']).__name__}\n"
            
            if 'y_pred' in data:
                if data.get('type') == 'classification':
                    accuracy = accuracy_score(data['y_test'], data['y_pred'])
                    info += f"🎯 Accuracy: {accuracy:.3f}\n"
                else:
                    mse = mean_squared_error(data['y_test'], data['y_pred'])
                    info += f"📉 MSE: {mse:.3f}\n"
        
        self.data_text.insert(1.0, info)
    
    def update_generated_code(self):
        """Update the generated code panel"""
        self.code_text.delete(1.0, tk.END)
        
        code = "# Generated ML Pipeline Code\n"
        code += "import numpy as np\n"
        code += "from sklearn.datasets import make_classification\n"
        code += "from sklearn.model_selection import train_test_split\n"
        code += "from sklearn.linear_model import LinearRegression\n"
        code += "from sklearn.tree import DecisionTreeClassifier\n\n"
        
        for i, block in enumerate(self.blocks):
            code += f"# Step {i+1}: {block.block_type}\n"
            
            if block.block_type == 'Generate Data':
                code += "X, y = make_classification(n_samples=100, n_features=2, random_state=42)\n"
            elif block.block_type == 'Train/Test Split':
                code += "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)\n"
            elif block.block_type == 'Linear Regression':
                code += "model = LinearRegression()\n"
                code += "model.fit(X_train, y_train)\n"
                code += "y_pred = model.predict(X_test)\n"
            elif block.block_type == 'Decision Tree':
                code += "model = DecisionTreeClassifier()\n"
                code += "model.fit(X_train, y_train)\n"
                code += "y_pred = model.predict(X_test)\n"
            
            code += "\n"
        
        self.code_text.insert(1.0, code)
    
    def stop_execution(self):
        """Stop pipeline execution"""
        self.is_running = False
        for block in self.blocks:
            block.unhighlight()
        self.log_message("⏹ Pipeline execution stopped", "warning")
    
    def reset_canvas(self):
        """Reset the canvas"""
        self.canvas.delete("all")
        self.blocks.clear()
        self.connections.clear()
        self.current_data = None
        self.is_running = False
        
        self.draw_grid()
        self.log_message("🔄 Canvas reset", "info")
        
        # Clear panels
        self.ax.clear()
        self.canvas_plot.draw()
        self.code_text.delete(1.0, tk.END)
        self.data_text.delete(1.0, tk.END)
    
    def save_pipeline(self):
        """Save the current pipeline"""
        if not self.blocks:
            messagebox.showwarning("Warning", "No pipeline to save!")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            pipeline_data = {
                'blocks': [{'type': b.block_type, 'x': b.x, 'y': b.y} for b in self.blocks]
            }
            
            with open(filename, 'w') as f:
                json.dump(pipeline_data, f, indent=2)
            
            self.log_message(f"💾 Pipeline saved to {filename}", "success")
    
    def load_pipeline(self):
        """Load a pipeline from file"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    pipeline_data = json.load(f)
                
                self.reset_canvas()
                
                for block_data in pipeline_data['blocks']:
                    block = MLBlock(self.canvas, block_data['type'], 
                                  block_data['x'], block_data['y'])
                    self.blocks.append(block)
                
                self.update_generated_code()
                self.log_message(f"📁 Pipeline loaded from {filename}", "success")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load pipeline: {str(e)}")
    
    def log_message(self, message, tag="info"):
        """Add message to log"""
        self.log_text.insert(tk.END, f"{message}\n", tag)
        self.log_text.see(tk.END)

def main():
    root = tk.Tk()
    app = MLVisualizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()