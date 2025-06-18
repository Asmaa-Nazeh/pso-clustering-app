## PSO Clustering Web Application

This is a simple web application built using Flask that performs clustering using the **Particle Swarm Optimization (PSO)** algorithm. Users can either upload their own dataset (CSV format) or generate random data for clustering.

---

### Features

* Upload CSV files containing numerical data.
* Configure clustering parameters:

  * Number of clusters
  * Number of particles (population size)
  * Maximum iterations
* Run PSO-based clustering on the input data.
* Visualize the results using:

  * A scatter plot of the clustered data and centroids.
  * A bar chart showing average values per cluster.

---

### Technologies Used

* Python 3
* Flask
* NumPy
* Pandas
* Matplotlib
* scikit-learn
* HTML and CSS

---

### How to Run the Application

1. **Clone or download the project:**

   ```bash
   git clone https://github.com/your-repo/pso-clustering.git
   cd pso-clustering
   ```

2. **Install the required dependencies:**

   ```bash
   pip install flask numpy pandas matplotlib scikit-learn
   ```

3. **Run the Flask application:**

   ```bash
   python appp.py
   ```

4. **Open your web browser** and go to:

   ```
   http://127.0.0.1:5000/
   ```

---

### Project Structure

```
├── appp.py           # Main application file with PSO and Flask logic
├── upload.html       # HTML form for user input
├── result.html       # HTML result page with plots
└── uploads/          # Folder to store uploaded CSV files (created automatically)
```

---

### PSO Parameters (Default Values)

* Number of particles: 30
* Number of iterations: 100
* Inertia weight (w): 0.5
* Cognitive coefficient (c1): 1.5
* Social coefficient (c2): 1.5

---

### Usage Notes

* The uploaded CSV file must contain only **numerical columns**. Non-numeric columns will be ignored.
* The application displays:

  * A scatter plot of clustered data points and cluster centers.
  * A bar chart showing average feature values per cluster.

---
