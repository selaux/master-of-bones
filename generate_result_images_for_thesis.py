from math import floor
import os
import vtk
import copy
import algorithms.comparison as cr
import helpers.loading as lh

DEFAULT = {
    'from_dir': 'data/2D/registered-outline/2015-06-08',
    'to_dir': 'thesis/img/results/parameter_comparison',
    'feature_extractor': cr.FeatureFlattenSplines,
    'window_extractor': cr.WindowExtractor2DByWindowLength,
    'step_size': 2,
    'window_size': 0.25,
    'number_of_evaluations': 20,
    'use_pca': True,
    'pca_components': 4,

    'metric': 6
}

evaluations = {
    'normal': [
        {
            'label': 'result',
            'values': {}
        }
    ],
    'window_size': [
        {
            'label': '0,05',
            'values': {
                'window_size': 0.05
            }
        },
        {
            'label': '0,1',
            'values': {
                'window_size': 0.1
            }
        },
        {
            'label': '0,2',
            'values': {
                'window_size': 0.2
            }
        },
        {
            'label': '0,4',
            'values': {
                'window_size': 0.4
            }
        },
        {
            'label': '0,8',
            'values': {
                'window_size': 0.8
            }
        },
        {
            'label': '1,6',
            'values': {
                'window_size': 1.6
            }
        }
    ],
    'window_extraction': [
        {
            'label': 'by-angle',
            'values': {
                'window_extractor': cr.WindowExtractor2DByAngle
            }
        },
        {
            'label': 'by-length',
            'values': {
                'window_extractor': cr.WindowExtractor2DByWindowLength
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
    ],
    'features': [
        {
            'label': 'Flattened Splines',
            'values': {
                'feature_extractor': cr.FeatureFlattenSplines
            }
        },
        {
            'label': 'Distance To Center',
            'values': {
                'feature_extractor': cr.FeatureDistanceToCenter
            }
        },
        {
            'label': 'Distance To Center and Curvature',
            'values': {
                'feature_extractor': cr.FeatureDistanceToCenterAndCurvature
            }
        },
        {
            'label': 'Curvature of Distance To Center',
            'values': {
                'feature_extractor': cr.FeatureCurvatureOfDistanceFromCenter
            }
        },
        {
            'label': 'Derivation of Mean Bone',
            'values': {
                'feature_extractor': cr.FeatureFlattenedDeviationVectorFromMeanBone
            }
        },
        {
            'label': 'Spline Derivatives',
            'values': {
                'feature_extractor': cr.FeatureSplineDerivatives
            }
        }
    ],
    'metric': [
        {
            'label': 'Mean Confidence',
            'values': {
                'metric': 0
            }
        },
        {
            'label': 'Mean Cross-Validation Accuracy',
            'values': {
                'metric': 1
            }
        },
        {
            'label': 'Mean Cross-Validation F1 Score',
            'values': {
                'metric': 2
            }
        },
        {
            'label': 'Margin',
            'values': {
                'metric': 3
            }
        },
        {
            'label': 'Observed Accuracy',
            'values': {
                'metric': 4
            }
        },
        {
            'label': 'Mean Cross-Validation Confidence',
            'values': {
                'metric': 6
            }
        }
    ],
    'missing-characteristic': [
        {
            'label': 'missing',
            'values': {
                'from_dir': 'data/2D/synthetic/2015-08-14-missing-characteristic',
                'to_dir': 'thesis/img/results/synthetic'
            }
        }
    ],
    'less-distinct': [
        {
            'label': 'less-distinct',
            'values': {
                'from_dir': 'data/2D/synthetic/2015-08-14-less-distinct-characteristic',
                'to_dir': 'thesis/img/results/synthetic'
            }
        }
    ],
    'wider': [
        {
            'label': 'wider',
            'values': {
                'from_dir': 'data/2D/synthetic/2015-08-14-wider-characteristic',
                'to_dir': 'thesis/img/results/synthetic'
            }
        }
    ],
    'shifted': [
        {
            'label': 'shifted',
            'values': {
                'from_dir': 'data/2D/synthetic/2015-08-14-shifted-characteristic',
                'to_dir': 'thesis/img/results/synthetic'
            }
        }
    ],
    'all': [
        {
            'label': 'all',
            'values': {
                'from_dir': 'data/2D/synthetic/2015-08-16-all',
                'to_dir': 'thesis/img/results/synthetic'
            }
        }
    ],
    'synthetic-errors-feat': [
        {
            'label': 'Features0',
            'values': {
                'from_dir': 'data/2D/synthetic/2015-08-16-all-feat-0',
                'to_dir': 'thesis/img/results/synthetic'
            }
        },
        {
            'label': 'Features1',
            'values': {
                'from_dir': 'data/2D/synthetic/2015-08-16-all-feat-1',
                'to_dir': 'thesis/img/results/synthetic'
            }
        },
        {
            'label': 'Features2',
            'values': {
                'from_dir': 'data/2D/synthetic/2015-08-16-all-feat-2',
                'to_dir': 'thesis/img/results/synthetic'
            }
        }
    ],
    'synthetic-errors-radius': [
        {
            'label': 'Radius0',
            'values': {
                'from_dir': 'data/2D/synthetic/2015-08-16-all-radius-0',
                'to_dir': 'thesis/img/results/synthetic'
            }
        },
        {
            'label': 'Radius1',
            'values': {
                'from_dir': 'data/2D/synthetic/2015-08-16-all-radius-1',
                'to_dir': 'thesis/img/results/synthetic'
            }
        },
        {
            'label': 'Radius2',
            'values': {
                'from_dir': 'data/2D/synthetic/2015-08-16-all-radius-2',
                'to_dir': 'thesis/img/results/synthetic'
            }
        }
    ],
    'synthetic-errors-trans': [
        {
            'label': 'Trans0',
            'values': {
                'from_dir': 'data/2D/synthetic/2015-08-16-all-trans-0',
                'to_dir': 'thesis/img/results/synthetic'
            }
        },
        {
            'label': 'Trans1',
            'values': {
                'from_dir': 'data/2D/synthetic/2015-08-16-all-trans-1',
                'to_dir': 'thesis/img/results/synthetic'
            }
        },
        {
            'label': 'Trans2',
            'values': {
                'from_dir': 'data/2D/synthetic/2015-08-16-all-trans-2',
                'to_dir': 'thesis/img/results/synthetic'
            }
        }
    ]
}
def get_metric_and_kwargs(observation):
    kwargs = copy.copy(DEFAULT)
    kwargs.update(observation['values'])
    metric = kwargs['metric']
    from_dir = kwargs['from_dir']
    to_dir = kwargs['to_dir']
    del kwargs['metric']
    del kwargs['from_dir']
    del kwargs['to_dir']

    return kwargs, metric, from_dir, to_dir

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

    tick_positions_x = vtk.vtkDoubleArray()
    tick_positions_y = vtk.vtkDoubleArray()
    for angle in [0, 90, 180, 270, 360]:
        tick_positions_x.InsertNextValue(angle)
    for height in [limit_min, floor(limit_max*10) / 10]:
        tick_positions_y.InsertNextValue(height)
    result.chart.GetAxis(0).SetRange(limit_min, limit_max)
    result.chart.GetAxis(0).SetCustomTickPositions(tick_positions_y)
    result.chart.GetAxis(1).SetRange(0, 360)
    result.chart.GetAxis(1).SetCustomTickPositions(tick_positions_x)
    result.chart.Update()

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
    for evaluation in evaluations:
        print("Evaluating Dimension {0}".format(evaluation))

        results_for_evaluation = []
        for i, observation in enumerate(evaluations[evaluation]):
            print("Evaluating {} of {}: {}".format(i+1, len(evaluations[evaluation]), observation['label']))

            kwargs, metric, from_dir, to_dir = get_metric_and_kwargs(observation)
            if not os.path.exists(os.path.join(to_dir, evaluation)):
                os.makedirs(os.path.join(to_dir, evaluation))
            loaded = lh.load_files(from_dir)
            classes = list(set(map(lambda b: b['class'], loaded)))
            result = cr.do_comparison(loaded, classes[0], classes[1], **kwargs)

            results_for_evaluation.append(result)

        for i, result in enumerate(results_for_evaluation):
            observation = evaluations[evaluation][i]
            kwargs, metric, from_dir, to_dir = get_metric_and_kwargs(observation)
            limit_min, limit_max = get_axis_limits(results_for_evaluation, metric)
            to_path = os.path.join(to_dir, evaluation, str(i) + '-' + observation['label'])
            render_chart(to_path, result, metric, limit_min, limit_max)

        print('Dimension Done')
    print('All Done!')

if __name__ == '__main__':
    main()
