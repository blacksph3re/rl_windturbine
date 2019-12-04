all:
	pdflatex paper
	bibtex paper
	pdflatex paper
	pdflatex paper

clean:
	rm *.aux *.out *.log *.bgl *.bbl *.toc *.pdf

watch:
	pdflatex -interaction=nonstopmode paper