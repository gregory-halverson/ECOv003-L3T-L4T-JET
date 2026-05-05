#!/bin/bash

# Script to render the ECOv003 L3 JET ATBD markdown to PDF
# Usage: ./render_pdf.sh

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define filenames
MARKDOWN_FILE="$SCRIPT_DIR/ECOv003_L3_L4_JET_ATBD.md"
PDF_FILE="$SCRIPT_DIR/ECOv003_L3_L4_JET_ATBD.pdf"
HEADER_FILE="$SCRIPT_DIR/pandoc_table_header.tex"

echo "Rendering ECOv003 L3 JET ATBD to PDF"
echo "Markdown file: $MARKDOWN_FILE"
echo "PDF file: $PDF_FILE"
echo "LaTeX header: $HEADER_FILE"

pandoc "$MARKDOWN_FILE" \
    -o "$PDF_FILE" \
    --pdf-engine=xelatex \
    --from=markdown+pipe_tables+grid_tables+table_captions \
    --include-in-header="$HEADER_FILE" \
    -V "mainfont:Arial Unicode MS" \
    -V geometry:margin=1in \
    -V colorlinks=true \
    -V linkcolor=blue \
    -V fontsize=11pt \
    -V linestretch=1.15

if [ $? -eq 0 ]; then
    echo "✅ PDF successfully generated: $PDF_FILE"
else
    echo "❌ Error generating PDF"
    exit 1
fi