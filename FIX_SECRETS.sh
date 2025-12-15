#!/bin/bash
# Script to remove secrets from git history

echo "================================================"
echo "Git Secret Removal Script"
echo "================================================"
echo ""
echo "⚠️  WARNING: This will rewrite git history!"
echo "   This is necessary to remove exposed secrets."
echo ""
echo "Files to remove from history:"
echo "  - deployment/.env"
echo "  - data/steps"
echo ""
read -p "Continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "Step 1: Removing deployment/.env from git history..."
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch deployment/.env' \
  --prune-empty --tag-name-filter cat -- --all

echo ""
echo "Step 2: Removing data/steps from git history..."
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch data/steps' \
  --prune-empty --tag-name-filter cat -- --all

echo ""
echo "Step 3: Cleaning up..."
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo ""
echo "================================================"
echo "✅ Secrets removed from git history!"
echo "================================================"
echo ""
echo "⚠️  IMPORTANT NEXT STEPS:"
echo ""
echo "1. ROTATE YOUR API KEYS (they're compromised!):"
echo "   - OpenAI: https://platform.openai.com/api-keys"
echo "   - AWS: AWS Console → IAM → Users"
echo ""
echo "2. Force push to GitHub:"
echo "   git push origin main --force"
echo ""
echo "3. Notify collaborators to re-clone (if any)"
echo ""
echo "4. Verify secrets are gone:"
echo "   git log --all --full-history -- deployment/.env"
echo "   git log --all --full-history -- data/steps"
echo "   (Should show nothing)"
echo ""
