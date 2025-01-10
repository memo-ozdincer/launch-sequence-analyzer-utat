import numpy as np
import pandas as pd
import torch
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import json
import threading
import time

class RealTimeMonitor:
    def __init__(self, model, scaler, window_size=10):
        self.model = model
        self.scaler = scaler
        self.window_size = window_size
        
        # Buffers for real-time data
        self.temp_buffer = deque(maxlen=window_size)
        self.pressure_buffer = deque(maxlen=window_size)
        self.vibration_buffer = deque(maxlen=window_size)
        
        # Buffers for plotting
        self.time_buffer = deque(maxlen=100)
        self.prediction_buffer = deque(maxlen=100)
        self.temp_plot_buffer = deque(maxlen=100)
        self.pressure_plot_buffer = deque(maxlen=100)
        self.vibration_plot_buffer = deque(maxlen=100)
        
        # Setup plotting
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.setup_plots()
        
        self.running = True
        
    def setup_plots(self):
        """Initialize the plots"""
        # Sensor data plot
        self.sensor_lines = []
        self.ax1.set_title('Real-time Sensor Data')
        self.ax1.set_ylabel('Normalized Values')
        self.ax1.grid(True)
        
        # Create lines for each sensor
        line_temp, = self.ax1.plot([], [], label='Temperature', color='red')
        line_pressure, = self.ax1.plot([], [], label='Pressure', color='blue')
        line_vibration, = self.ax1.plot([], [], label='Vibration', color='green')
        self.sensor_lines.extend([line_temp, line_pressure, line_vibration])
        self.ax1.legend()
        
        # Prediction probability plot
        self.ax2.set_title('Anomaly Prediction Probability')
        self.ax2.set_ylabel('Probability')
        self.ax2.set_ylim(0, 1)
        self.ax2.grid(True)
        self.prediction_line, = self.ax2.plot([], [], color='purple')
        
        # Add warning threshold line
        self.ax2.axhline(y=0.7, color='r', linestyle='--', alpha=0.5)
        
    def process_window(self):
        """Process current window of sensor data"""
        if len(self.temp_buffer) < self.window_size:
            return None
            
        # Extract features from buffers
        temp_stats = [np.mean(self.temp_buffer), np.std(self.temp_buffer)]
        pressure_stats = [np.mean(self.pressure_buffer), np.std(self.pressure_buffer)]
        vibration_stats = [np.mean(self.vibration_buffer), np.std(self.vibration_buffer)]
        
        # Combine features
        features = np.array(temp_stats + pressure_stats + vibration_stats).reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            prediction = torch.softmax(
                self.model(torch.FloatTensor(features_scaled)), 
                dim=1
            ).numpy()[0]
            
        return prediction[1]  # Return probability of anomaly
        
    def update_plot(self, frame):
        """Update function for animation"""
        # Update sensor lines
        for line, data in zip(self.sensor_lines, 
                            [self.temp_plot_buffer, 
                             self.pressure_plot_buffer, 
                             self.vibration_plot_buffer]):
            line.set_data(range(len(data)), list(data))
            
        # Update prediction line
        self.prediction_line.set_data(range(len(self.prediction_buffer)), 
                                    list(self.prediction_buffer))
        
        # Adjust x-axis limits
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view(scalex=True, scaley=False)
        
        return self.sensor_lines + [self.prediction_line]
        
    def simulate_sensor_data(self):
        """Simulate incoming sensor data"""
        while self.running:
            # Simulate sensor readings
            timestamp = datetime.now()
            temperature = np.random.normal(350, 10)
            pressure = np.random.normal(1000, 50)
            vibration = np.random.normal(0.5, 0.1)
            
            # Add to buffers
            self.temp_buffer.append(temperature)
            self.pressure_buffer.append(pressure)
            self.vibration_buffer.append(vibration)
            
            # Make prediction if enough data
            if len(self.temp_buffer) == self.window_size:
                prediction = self.process_window()
                if prediction is not None:
                    self.time_buffer.append(timestamp)
                    self.prediction_buffer.append(prediction)
                    self.temp_plot_buffer.append(temperature)
                    self.pressure_plot_buffer.append(pressure)
                    self.vibration_plot_buffer.append(vibration)
                    
                    # Log warning if probability is high
                    if prediction > 0.7:
                        print(f"WARNING: High anomaly probability detected: {prediction:.2f}")
                        
            time.sleep(0.1)  # Simulate 10Hz data collection
            
    def run(self):
        """Run the monitor"""
        # Start data simulation in separate thread
        sensor_thread = threading.Thread(target=self.simulate_sensor_data)
        sensor_thread.start()
        
        # Create animation
        ani = FuncAnimation(self.fig, self.update_plot, interval=100,
                          blit=True)
        plt.show()
        
        # Cleanup
        self.running = False
        sensor_thread.join()

def main():
    # Load trained model and scaler
    model = torch.load('engine_model.pth')
    scaler = pd.read_pickle('scaler.pkl')
    
    # Create and run monitor
    monitor = RealTimeMonitor(model, scaler)
    monitor.run()

if __name__ == "__main__":
    main()