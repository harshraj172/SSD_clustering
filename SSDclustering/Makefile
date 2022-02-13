# Makefile

NAME     = mk_template
BIBTEX   = bibtex
PDFLATEX = pdflatex
RM     = rm -f
CRUFT  = $(NAME).aux $(NAME).bbl $(NAME).blg $(NAME).dvi $(NAME).log \
		 $(NAME).lof $(NAME).lot $(NAME).toc \
		 $(NAME).pdf $(NAME).ps $(NAME).log $(NAME).out
INPUTS = $(NAME).tex my-refs.bib

all: $(NAME).pdf

$(NAME).aux: $(INPUTS)
	$(PDFLATEX) '\scrollmode \input '"$<"

$(NAME).bbl: $(NAME).aux
	$(BIBTEX) $(NAME).aux

$(NAME).pdf: $(NAME).aux $(NAME).bbl
	$(PDFLATEX) $(NAME).tex
	$(PDFLATEX) $(NAME).tex

clean:
	$(RM) $(CRUFT) $(NAME).pdf
