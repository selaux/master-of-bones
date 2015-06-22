all: dbsma.pdf

dbsma.pdf:
	pdflatex dbsma
	bibtex dbsma
	pdflatex dbsma
	pdflatex dbsma

clean:
	rm -f *~ *.bak *.aux {dbsba,dbsma}.{log,toc,lof,lot,blg,bbl,pdf}
