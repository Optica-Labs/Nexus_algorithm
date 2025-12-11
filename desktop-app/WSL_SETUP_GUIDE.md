# WSL Setup Guide - Vector Precognition Desktop

**Issue**: Electron desktop apps don't run well in WSL because WSL doesn't have full GUI support by default.

---

## 🎯 **Recommended Solutions**

### **Option 1: Run on Windows (Recommended)**

The desktop app is meant for native Windows/Mac/Linux, not WSL.

#### Steps:
1. **Open Windows PowerShell** (not WSL terminal)
   - Press `Win + X` → Choose "Windows PowerShell"

2. **Navigate to project** (adjust path):
   ```powershell
   cd C:\Users\<YourUsername>\wsl$\Ubuntu\home\aya\work\optica_labs\algorithm_work\desktop-app
   ```

   Or clone the repo directly in Windows:
   ```powershell
   cd C:\Users\<YourUsername>\Documents
   git clone <repo-url>
   cd algorithm_work\desktop-app
   ```

3. **Run setup**:
   ```powershell
   .\setup.ps1
   ```

4. **Launch app**:
   ```powershell
   .\start.ps1
   ```

---

### **Option 2: Test Backend Only (WSL)**

Test the Python backend without Electron GUI:

```bash
# In WSL terminal
cd ~/work/optica_labs/algorithm_work/desktop-app
./test_backend.sh
```

Then open your **Windows browser** to: `http://localhost:8501`

This lets you test ChatGPT integration without Electron!

---

### **Option 3: Install X Server for WSL (Advanced)**

Make WSL support GUI apps:

#### 1. Install VcXsrv (Windows X Server)
- Download: [sourceforge.net/projects/vcxsrv](https://sourceforge.net/projects/vcxsrv/)
- Install and run "XLaunch"
- Choose: "Multiple windows" → "Start no client" → **Disable access control**

#### 2. Configure WSL
```bash
# Add to ~/.bashrc
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
export LIBGL_ALWAYS_INDIRECT=1
```

#### 3. Install Linux libraries
```bash
sudo apt update
sudo apt install -y \
  libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
  libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 \
  libxfixes3 libxrandr2 libgbm1 libasound2 \
  libatspi2.0-0 libxshmfence1 libgtk-3-0
```

#### 4. Try launching
```bash
cd ~/work/optica_labs/algorithm_work/desktop-app
./start.sh
```

**Note**: This is complex and may have issues. Option 1 or 2 is better!

---

## 🧪 **Quick Test (Backend Only)**

### Test without Electron:

```bash
# Start backend server
cd ~/work/optica_labs/algorithm_work/desktop-app
./test_backend.sh
```

### What you can test:
✅ API key configuration screen
✅ ChatGPT integration
✅ Real-time risk metrics
✅ Live charts
✅ Safety alerts
✅ Conversation export

**Only difference**: No native desktop window, runs in browser instead.

---

## 📋 **Comparison**

| Feature | Windows Native | WSL + X Server | Backend Only (Browser) |
|---------|----------------|----------------|------------------------|
| **Ease of Setup** | ⭐⭐⭐ Easy | ⭐ Hard | ⭐⭐⭐ Easy |
| **Functionality** | 100% | 90% | 95% |
| **Performance** | Excellent | Fair | Good |
| **API Key Storage** | Encrypted | Encrypted | Session only |
| **Desktop Icon** | ✅ Yes | ✅ Yes | ❌ No |
| **Installers** | ✅ Yes | ❌ No | ❌ No |

---

## 🚀 **Quick Start Commands**

### For Windows (PowerShell):
```powershell
cd desktop-app
.\setup.ps1
.\start.ps1
```

### For WSL (Backend Only):
```bash
cd ~/work/optica_labs/algorithm_work/desktop-app
./test_backend.sh
# Open browser to http://localhost:8501
```

### For WSL (With X Server):
```bash
# Install X server on Windows first
sudo apt install -y libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2
./start.sh
```

---

## 🐛 **Troubleshooting**

### "libnss3.so: cannot open shared object file"
→ Missing libraries. Install with:
```bash
sudo apt install -y libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2
```

### "Cannot connect to X server"
→ X server not running on Windows. Install VcXsrv and start it.

### "DISPLAY not set"
→ Add to ~/.bashrc:
```bash
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
```

### Backend runs but Electron window is blank
→ Streamlit is starting up. Wait 30 seconds, or check terminal for errors.

---

## 💡 **Recommendation**

**For best experience**: Use **Option 1 (Windows Native)** or **Option 2 (Backend Only)**.

Option 3 (X Server) is only needed if you specifically want the native desktop window in WSL, which is rarely necessary for testing.

---

## 📞 **Need Help?**

1. Try backend-only mode first: `./test_backend.sh`
2. If that works, ChatGPT integration is confirmed working!
3. For desktop window, switch to Windows PowerShell
4. Check [README.md](README.md) for full documentation

---

**Current Environment**: WSL (Windows Subsystem for Linux)
**Detected Issue**: Electron GUI libraries missing
**Best Solution**: Run on Windows natively OR test backend in browser
