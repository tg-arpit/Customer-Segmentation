from flask import Flask, render_template, request
from analysis import load_data, get_data_overview, plot_eda, elbow_and_silhouette, run_kmeans

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/step1')
def step1():
    df = load_data()
    info = get_data_overview(df)
    return render_template('step1.html', info=info)

@app.route('/step2')
def step2():
    df = load_data()
    charts = plot_eda(df)
    return render_template('step2.html', charts=charts)

@app.route('/step3')
def step3():
    df = load_data()
    chart, best_k, best_sil = elbow_and_silhouette(df)
    return render_template('step3.html', chart=chart, best_k=best_k, best_sil=best_sil)

@app.route('/step4', methods=['GET', 'POST'])
def step4():
    df = load_data()
    k = int(request.form.get('k', 5))
    results = run_kmeans(df, k)
    results['k'] = k
    return render_template('step4.html', results=results)

@app.route('/step5')
def step5():
    return render_template('step5.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
