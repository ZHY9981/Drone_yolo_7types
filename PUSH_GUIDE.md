# Git Push 操作指南 — aerial-yolo26s-drone-detection

## 前置条件
1. 在 GitHub 网页先建好仓库：https://github.com/new
   - Repository name: `aerial-yolo26s-drone-detection`
   - Public, MIT License
   - ⚠️ 不要勾 "Add a README file"
   - 点 Create repository

2. 准备好 Personal Access Token（PAT）
   - GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic) → Generate new token
   - 勾 `repo` 权限，生成后复制下来（只显示一次！）

## 推送步骤

打开 Git Bash 或 PowerShell，逐条执行：

```bash
# 1. 进入仓库目录
cd "C:/Users/ZHY/WorkBuddy/2026-07-28-18-27-55/aerial-yolo26s-drone-detection"

# 2. 初始化并配置
git init
git branch -M main
git remote add origin https://github.com/ZHY9981/aerial-yolo26s-drone-detection.git

# 3. 添加所有文件并提交
git add .
git commit -m "Initial commit: YOLO26s aerial small-object detection (20+ ablation runs)"

# 4. 推送
git push -u origin main
```

## 认证说明
`git push` 时会弹窗口或命令行提示输入：
- 用户名: `ZHY9981`
- 密码: **粘贴刚才复制的 PAT**（不是 GitHub 登录密码！）

## 推送后检查
- 打开 https://github.com/ZHY9981/aerial-yolo26s-drone-detection
- 确认 README 能正常显示
- Repository 右上角 Settings → 拉到下面：确认仓库是 Public
- About 区域填简介：填 `yolo / drone / computer-vision / small-object` 等 topics

## 如果推送前要检查文件结构
```bash
dir /b /s
```
应该看到 10 个文件在正确位置。
