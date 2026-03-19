#!/bin/bash
# =============================================
# Check size of all staged files before commit
# =============================================

LIMIT_MB=10
LIMIT_BYTES=$((LIMIT_MB * 1024 * 1024))

echo "================================================"
echo "📋 STAGED FILES SIZE REPORT (Limit: ${LIMIT_MB}MB)"
echo "================================================"

LARGE_FILES=()
SAFE_FILES=()
TOTAL_SIZE=0

while IFS= read -r file; do
    if [ -f "$file" ]; then
        size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null)
        size_mb=$(echo "scale=2; $size / 1024 / 1024" | bc)
        TOTAL_SIZE=$((TOTAL_SIZE + size))

        if [ "$size" -gt "$LIMIT_BYTES" ]; then
            LARGE_FILES+=("$file | ${size_mb}MB")
        else
            SAFE_FILES+=("$file | ${size_mb}MB")
        fi
    fi
done < <(git diff --cached --name-only --diff-filter=ACM)

# Print safe files
echo ""
echo "✅ SAFE FILES (< ${LIMIT_MB}MB):"
echo "------------------------------------------------"
if [ ${#SAFE_FILES[@]} -eq 0 ]; then
    echo "   (none)"
else
    for f in "${SAFE_FILES[@]}"; do
        echo "   ✅ $f"
    done
fi

# Print large files
echo ""
echo "❌ LARGE FILES (> ${LIMIT_MB}MB) — SHOULD NOT PUSH:"
echo "------------------------------------------------"
if [ ${#LARGE_FILES[@]} -eq 0 ]; then
    echo "   (none) — All clear!"
else
    for f in "${LARGE_FILES[@]}"; do
        echo "   ❌ $f"
    done
fi

# Print total
echo ""
echo "================================================"
TOTAL_MB=$(echo "scale=2; $TOTAL_SIZE / 1024 / 1024" | bc)
echo "📦 Total staged size : ${TOTAL_MB}MB"
echo "🚨 Large files count : ${#LARGE_FILES[@]}"
echo "✅ Safe files count  : ${#SAFE_FILES[@]}"
echo "================================================"

# Suggest fix if large files found
if [ ${#LARGE_FILES[@]} -gt 0 ]; then
    echo ""
    echo "💡 Run these commands to fix:"
    echo "------------------------------------------------"
    for entry in "${LARGE_FILES[@]}"; do
        file=$(echo "$entry" | cut -d'|' -f1 | xargs)
        echo "   git rm -r --cached \"$file\""
        echo "   echo \"$file\" >> .gitignore"
    done
    echo "   git add .gitignore"
    echo "   git commit -m \"Organize files\""
fi
