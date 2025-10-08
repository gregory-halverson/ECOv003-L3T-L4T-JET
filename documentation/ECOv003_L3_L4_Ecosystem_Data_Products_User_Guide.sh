#!/bin/bash

# Script to render the ECOSTRESS User Guide markdown to PDF
# Usage: ./render_pdf.sh

echo "Rendering ECOSTRESS User Guide to PDF..."

pandoc "ECOv003_L3_L4_Ecosystem_Data_Products_User_Guide.md" \
    -o "ECOv003_L3_L4_Ecosystem_Data_Products_User_Guide.pdf" \
    --pdf-engine=xelatex \
    -V "mainfont:Arial Unicode MS" \
    -V geometry:margin=1in \
    -V colorlinks=true \
    -V linkcolor=blue \
    -V fontsize=11pt \
    -V linestretch=1.15

if [ $? -eq 0 ]; then
    echo "✅ PDF successfully generated: ECOv003_L3_L4_Ecosystem_Data_Products_User_Guide.pdf"
else
    echo "❌ Error generating PDF"
    exit 1
fi