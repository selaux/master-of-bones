import os
import vtk
import copy
import algorithms.comparison as cr
import helpers.loading as lh

FROM_DIR = 'data/2D/registered-outline/2015-06-08'
TO_DIR = 'thesis/img/results/parameter_comparison'
DEFAULT = {
    'feature_extractor': cr.FeatureFlattenSplines,
    'window_extractor': cr.WindowExtractor2DByWindowLength,
    'metric': 0,
    'step_size': 1,
    'window_size': 0.2,
    'number_of_evaluations': 20,
    'use_pca': True,
    'pca_components': 4,

    'metric': 1
}
evaluations = {
    'window_size': [
        {
            'label': '0.05',
            'values': {
                'window_size': 0.05
            }
        },
        {
            'label': '0.1',
            'values': {
                'window_size': 0.1
            }
        },
        {
            'label': '0.2',
            'values': {
                'window_size': 0.2
            }
        },
        {
            'label': '0.4',
            'values': {
                'window_size': 0.4
            }
        },
        {
            'label': '0.8',
            'values': {
                'window_size': 0.8
            }
        },
        {
            'label': '1.6',
            'values': {
                'window_size': 1.6
            }
        }
    ],
    'number_of_evaluations': [
        {
            'label': '5',
            'values': {
                'number_of_evaluations': 5
            }
        },
        {
            'label': '10',
            'values': {
                'number_of_evaluations': 10
            }
        },
        {
            'label': '15',
            'values': {
                'number_of_evaluations': 15
            }
        },
        {
            'label': '20',
            'values': {
                'number_of_evaluations': 20
            }
        },
        {
            'label': '25',
            'values': {
                'number_of_evaluations': 25
            }
        },
        {
            'label': '50',
            'values': {
                'number_of_evaluations': 50
            }
        }
    ],
    'pca': [
        {
            'label': 'No PCA',
            'values': {
                'use_pca': False
            }
        },
        {
            'label': '1 PCA Component',
            'values': {
                'pca_components': 1
            }
        },
        {
            'label': '2 PCA Components',
            'values': {
                'pca_components': 2
            }
        },
        {
            'label': '3 PCA Components',
            'values': {
                'pca_components': 3
            }
        },
        {
            'label': '4 PCA Components',
            'values': {
                'pca_components': 4
            }
        },
        {
            'label': '5 PCA Components',
            'values': {
                'pca_components': 5
            }
        }
    ]
}

def get_metric_and_kwargs(observation):
    kwargs = copy.copy(DEFAULT)
    kwargs.update(observation['values'])
    metric = kwargs['metric']
    del kwargs['metric']

    return kwargs, metric

def get_axis_limits(results, metric):
    metrics = []

    for r in results:
        metrics += [s.get_performance_indicators()[metric]['value'] for s in r.single_results]

    max_value = max(metrics)
    min_value = min(metrics)
    padding = 0.1 * max_value

    max_value += padding
    min_value -= padding
    min_value = min_value if min_value >= 0 else 0

    return min_value, max_value

def render_chart(filename, result, metric, limit_min, limit_max):
    result.chart.GetAxis(0).SetGridVisible(False)
    result.chart.GetAxis(1).SetGridVisible(False)
    result.chart_line.SetWidth(5)

    result.update_actor(0.5, metric)

    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.SetSize(640, 360)

    result.chart.GetAxis(0).SetRange(limit_min, limit_max)

    graph_context = vtk.vtkContextView()
    graph_context.SetRenderWindow(render_window)

    graph_context.GetScene().AddItem(result.chart)

    exporter = vtk.vtkGL2PSExporter()
    exporter.SetRenderWindow(render_window)
    exporter.SetFileFormatToSVG()
    exporter.CompressOff()
    exporter.DrawBackgroundOff()
    exporter.SetFilePrefix(filename)
    exporter.Write()

    render_window.Finalize()

def main():
    loaded = lh.load_files(FROM_DIR)
    classes = list(set(map(lambda b: b['class'], loaded)))
    for evaluation in evaluations:
        print("Evaluating Dimension {0}".format(evaluation))

        if not os.path.exists(os.path.join(TO_DIR, evaluation)):
            os.makedirs(os.path.join(TO_DIR, evaluation))

        results_for_evaluation = []
        for i, observation in enumerate(evaluations[evaluation]):
            print("Evaluating {} of {}: {}".format(i+1, len(evaluations[evaluation]), observation['label']))

            kwargs, metric = get_metric_and_kwargs(observation)
            result = cr.do_comparison(loaded, classes[0], classes[1], **kwargs)

            results_for_evaluation.append(result)

        for i, result in enumerate(results_for_evaluation):
            observation = evaluations[evaluation][i]
            kwargs, metric = get_metric_and_kwargs(observation)
            limit_min, limit_max = get_axis_limits(results_for_evaluation, metric)
            to_path = os.path.join(TO_DIR, evaluation, str(i) + '-' + observation['label'])
            render_chart(to_path, result, metric, limit_min, limit_max)

        print('Dimension Done')
    print('All Done!')

if __name__ == '__main__':
    main()
