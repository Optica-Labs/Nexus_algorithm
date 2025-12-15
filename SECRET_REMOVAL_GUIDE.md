# Secret Removal Guide - Fix GitHub Push Protection

**Issue**: GitHub blocked your push due to exposed API keys in git history.

**Files containing secrets**:
- `deployment/.env` - AWS keys
- `data/steps` - OpenAI API key

---

## 🚨 STEP 1: ROTATE API KEYS (DO THIS FIRST!)

Since these keys are in git history (even though you'll remove them), they must be considered **compromised**.

### OpenAI API Key

1. Go to: https://platform.openai.com/api-keys
2. Find the exposed key
3. Click "Revoke" or "Delete"
4. Create a new API key
5. Save it securely (NOT in git!)

### AWS Access Keys

1. Go to: AWS Console → IAM → Users → Your User
2. Click "Security credentials"
3. Find the exposed Access Key ID
4. Click "Make inactive" then "Delete"
5. Click "Create access key"
6. Download and save securely (NOT in git!)

---

## 🔧 STEP 2: Remove Secrets from Git History

You have **3 options**:

### Option A: Use BFG Repo Cleaner (Easiest) ⭐

```bash
# Install BFG
# On Ubuntu/WSL:
sudo apt-get install bfg

# On Mac:
brew install bfg

# Remove the files from all history:
cd /home/aya/work/optica_labs/algorithm_work
bfg --delete-files .env
bfg --delete-files steps

# Clean up
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

### Option B: Use git filter-repo (Recommended)

```bash
# Install git-filter-repo
pip install git-filter-repo

# Remove the files
cd /home/aya/work/optica_labs/algorithm_work
git filter-repo --path deployment/.env --invert-paths
git filter-repo --path data/steps --invert-paths
```

### Option C: Manual with git filter-branch

```bash
cd /home/aya/work/optica_labs/algorithm_work

# Remove deployment/.env
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch deployment/.env' \
  --prune-empty --tag-name-filter cat -- --all

# Remove data/steps
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch data/steps' \
  --prune-empty --tag-name-filter cat -- --all

# Clean up
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

---

## ✅ STEP 3: Verify Secrets Are Gone

```bash
# Check if files still exist in history
git log --all --full-history -- deployment/.env
git log --all --full-history -- data/steps

# Should return nothing!
```

---

## 📤 STEP 4: Force Push to GitHub

⚠️ **Warning**: This rewrites history. Collaborators will need to re-clone.

```bash
git push origin main --force
```

---

## 🛡️ STEP 5: Prevent Future Exposures

Your `.gitignore` already has these files listed, but make sure they're actually ignored:

```bash
# Verify files are ignored
git check-ignore deployment/.env data/steps

# Should return:
# deployment/.env
# data/steps
```

If not ignored, add them:

```bash
# Add to .gitignore
echo "deployment/.env" >> .gitignore
echo "data/steps" >> .gitignore

git add .gitignore
git commit -m "chore: Ensure secret files are in .gitignore"
```

---

## 🔐 STEP 6: Use Environment Variables Instead

### For deployment/.env

**Never commit this file again!**

Instead, use:

1. **Local development**: Keep `.env` local (already in `.gitignore`)
2. **CI/CD**: Use GitHub Secrets
3. **Production**: Use environment variables on server

**Example `.env.example` (commit this instead)**:

```bash
# .env.example - Template for environment variables
# Copy to .env and fill in your values

AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here
AWS_DEFAULT_REGION=us-east-1
OPENAI_API_KEY=your_openai_key_here
```

### For data/steps

If this contains API keys, **don't commit it**.

Create a template instead:

```bash
# Move the real file to a safe location
mv data/steps data/steps.local

# Create a template
cat > data/steps.example << 'EOF'
# steps.example
# Copy to 'steps' and add your actual keys

Step 1: Get API key from...
Step 2: Configure...
EOF

# Commit the template
git add data/steps.example
git commit -m "docs: Add steps template without secrets"
```

---

## 📋 Quick Reference

### Fastest Solution (Copy & Paste):

```bash
# 1. Install BFG (if not installed)
sudo apt-get install bfg -y

# 2. Remove secrets from history
cd /home/aya/work/optica_labs/algorithm_work
bfg --delete-files .env
bfg --delete-files steps

# 3. Clean up git
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 4. Verify they're gone
git log --all --full-history -- deployment/.env
git log --all --full-history -- data/steps

# 5. Force push
git push origin main --force
```

### If BFG Not Available:

```bash
# Use the FIX_SECRETS.sh script I created:
./FIX_SECRETS.sh

# Follow the prompts
```

---

## ⚠️ Important Warnings

1. **Force push rewrites history** - Collaborators need to know!
2. **Rotate keys FIRST** - The exposed keys are compromised
3. **Don't commit secrets again** - Use `.env.example` templates
4. **Notify your team** - If you're working with others

---

## 🆘 If Something Goes Wrong

### Backup First!

```bash
# Create a backup before cleaning
cd /home/aya/work/optica_labs/algorithm_work
git bundle create ../repo-backup.bundle --all
```

### Restore from Backup:

```bash
git clone ../repo-backup.bundle restored-repo
cd restored-repo
```

---

## ✅ Checklist

- [ ] **CRITICAL**: Rotated OpenAI API key
- [ ] **CRITICAL**: Rotated AWS access keys
- [ ] Removed secrets from git history (Option A, B, or C)
- [ ] Verified secrets are gone (`git log` shows nothing)
- [ ] Force pushed to GitHub
- [ ] Created `.env.example` template
- [ ] Created `data/steps.example` template
- [ ] Verified `.gitignore` includes secret files
- [ ] Updated local `.env` with new keys
- [ ] Tested that app still works with new keys

---

## 📞 Additional Resources

- **GitHub Secret Scanning**: https://docs.github.com/en/code-security/secret-scanning
- **BFG Repo Cleaner**: https://rtyley.github.io/bfg-repo-cleaner/
- **git-filter-repo**: https://github.com/newren/git-filter-repo
- **Removing Sensitive Data**: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository

---

**Status**: Choose an option above and follow the steps!
**Estimated Time**: 10-15 minutes
**Difficulty**: Medium (but necessary!)
