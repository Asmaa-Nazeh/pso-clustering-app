from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.metrics import pairwise_distances_argmin
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

#  PSO Parameters 
n_particles = 30
n_iterations = 100
w = 0.5
c1 = 1.5
c2 = 1.5

#  PSO Clustering Function
class Particle:
    def __init__(self, n_clusters, X):
        self.position = X[np.random.choice(range(X.shape[0]), n_clusters, replace=False)].astype(np.float64)
        self.velocity = np.zeros_like(self.position)
        self.best_position = np.copy(self.position)
        self.best_score = float('inf')

    def evaluate(self, X):
        labels = pairwise_distances_argmin(X, self.position)
        score = 0.0
        for i in range(len(self.position)):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                score += np.sum((cluster_points - self.position[i]) ** 2)
        return score, labels

def pso_clustering(X, n_clusters):
    X = X.astype(np.float64)
    particles = [Particle(n_clusters, X) for _ in range(n_particles)]
    global_best_position = None
    global_best_score = float('inf')

    for _ in range(n_iterations):
        for particle in particles:
            score, _ = particle.evaluate(X)
            if score < particle.best_score:
                particle.best_score = score
                particle.best_position = np.copy(particle.position)
            if score < global_best_score:
                global_best_score = score
                global_best_position = np.copy(particle.position)

        for particle in particles:
            r1, r2 = np.random.rand(), np.random.rand()
            cognitive = c1 * r1 * (particle.best_position - particle.position)
            social = c2 * r2 * (global_best_position - particle.position)
            particle.velocity = w * particle.velocity + cognitive + social
            particle.position += particle.velocity

    final_labels = pairwise_distances_argmin(X, global_best_position)
    return X, final_labels, global_best_position

#  Flask Routes
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        try:
            n_clusters = int(request.form.get('n_clusters', 4))
        except (ValueError, TypeError):
            n_clusters = 4
        print(f"[DEBUG] Received number of clusters from form: {n_clusters}")

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            
            df = pd.read_csv(filepath)

            
            numeric_columns = df.select_dtypes(include=[float, int]).columns
            df_numeric = df[numeric_columns]  

            
            X = df_numeric.values.astype(np.float64)

            
            data, labels, centers = pso_clustering(X, n_clusters)

            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=30)
            ax1.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X')
            ax1.set_title(f"Customer Clustering using PSO (k = {n_clusters})")
            ax1.set_xlabel("Feature 1")
            ax1.set_ylabel("Feature 2")

           
            df['Cluster'] = labels
            grouped = df.groupby('Cluster').mean()
            grouped.plot(kind='bar', ax=ax2, color=['skyblue', 'lightgreen'])
            ax2.set_title('Average Values per Cluster')
            ax2.set_ylabel('Value')
            ax2.set_xlabel('Cluster')
            ax2.legend(grouped.columns)

            
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plot_data = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            plt.close()

            return render_template('result.html', plot_url=plot_data)

    return render_template('upload.html')

#  Run App 
if __name__ == '__main__':
    app.run(debug=True)
