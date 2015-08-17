'use strict';

var gulp = require('gulp'),
    using = require('gulp-using'),
    through2 = require('through2'),
    spawn = require('child_process').spawn,
    path = require('path'),
    cheerio = require('cheerio');

function lineWidth(to) {
    return through2.obj(function(file, encoding, done) {
        var $ = cheerio.load(file.contents.toString('utf8'), {
            xmlMode: true
        });
        console.log('attr: "' + $('polyline').attr('stroke-width') + '"')
        if ($('polyline').attr('stroke-width')) {
            $('polyline').attr('stroke-width', to);
        }

        file.contents = new Buffer($.html());
        this.push(file);
        done();
    });
}

function inkscape() {
    return through2.obj(function(file, encoding, done) {
        var dirname = path.dirname(file.path),
            basename = path.basename(file.path, '.svg'),
            childProcess = spawn('inkscape', [
                    '-D',
                    '-z',
                    '--file=' + file.path,
                    '--export-pdf=' + dirname + '/' + basename + '.pdf',
                    //'--export-latex'
            ]);

        childProcess.on('error', function (e) {
            done(e);
        });
        childProcess.on('close', function () {
            done();
        });
    });
}

gulp.task('updateCharts', function() {
    gulp.src('img/results/**/*.svg')
        .pipe(using({}))
        .pipe(lineWidth("5.0"))
        .pipe(gulp.dest('img/results'))
});

gulp.task('svgToPdf', [ 'updateCharts' ], function() {
    gulp.src('img/**/*.svg')
        .pipe(using({}))
        .pipe(inkscape())
});

gulp.task('default', ['updateCharts', 'svgToPdf']);
