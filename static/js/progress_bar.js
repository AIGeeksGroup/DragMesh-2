// document.addEventListener('DOMContentLoaded', function () {
//     const globe = document.querySelector('.globe');
//     const wrapper = globe.querySelector('.wrapper');
//     const container = document.querySelector('.section-path');
  
//     const video1 = document.getElementById('video1');
//     const video2 = document.getElementById('video2');
  
//     let isDragging = false;
//     let offsetX = 0;
  
//     const containerRect = container.getBoundingClientRect();
//     const globeRect = globe.getBoundingClientRect();
  
//     const edgePadding = 10;
//     const maxLeft = containerRect.width - globeRect.width - edgePadding;
//     const minLeft = 0;
  
//     function startDrag(e) {
//       isDragging = true;
//       globe.style.cursor = 'grabbing';
  
//       const clientX = e.type.includes('touch') ? e.touches[0].clientX : e.clientX;
//       offsetX = clientX - globe.offsetLeft;
  
//       globe.style.transition = 'none';
//       wrapper.style.transition = 'none';
  
//       // 可选：拖动时暂停两个视频
//       video1.pause();
//       video2.pause();
//     }
  
//     function onDrag(e) {
//       if (!isDragging) return;
  
//       const clientX = e.type.includes('touch') ? e.touches[0].clientX : e.clientX;
//       let newLeft = clientX - offsetX;
  
//       newLeft = Math.max(minLeft, Math.min(newLeft, maxLeft));
//       globe.style.left = newLeft + 'px';
  
//       const percent = (newLeft - minLeft) / (maxLeft - minLeft);
//       const wrapperLeft = -462 + percent * 462;
//       wrapper.style.left = `${wrapperLeft}px`;
  
//       // ✅ 同步控制两个视频的播放进度
//       if (video1.duration) {
//         video1.currentTime = percent * video1.duration;
//       }
//       if (video2.duration) {
//         video2.currentTime = percent * video2.duration;
//       }
//     }
  
//     function stopDrag() {
//       isDragging = false;
//       globe.style.cursor = 'grab';
  
//       // 可选：拖动结束后继续播放两个视频
//       // video1.play();
//       // video2.play();
//     }
  
//     globe.addEventListener('mousedown', startDrag);
//     globe.addEventListener('touchstart', startDrag, { passive: true });
  
//     document.addEventListener('mousemove', onDrag);
//     document.addEventListener('touchmove', onDrag, { passive: false });
  
//     document.addEventListener('mouseup', stopDrag);
//     document.addEventListener('touchend', stopDrag);
//   });


document.addEventListener('DOMContentLoaded', function () {
    const globe = document.querySelector('.globe');
    const wrapper = globe.querySelector('.wrapper');
    const container = document.querySelector('.section-path');
  
    const videoLeft = document.getElementById('videoLeft');
    const videoRight = document.getElementById('videoRight');
  
    // ✅ 视频组列表，每组包含一对视频
    const videoGroups = [
      { left: 'static/videos/scene_0000/realscan.mkv', right: 'static/videos/scene_0000/synthetic.mkv' },
      { left: 'static/videos/scene_0006/realscan.mkv', right: 'static/videos/scene_0006/synthetic.mkv' },
      { left: 'static/videos/scene_0083/realscan.mkv', right: 'static/videos/scene_0083/synthetic.mkv' }
    ];
  
    let currentGroupIndex = 0;
    let currentPercent = 0;
  
    // ✅ 初始化当前视频组
    function loadCurrentGroup() {
      const group = videoGroups[currentGroupIndex];
      videoLeft.src = group.left;
      videoRight.src = group.right;
  
      videoLeft.load();
      videoRight.load();
  
      // 等待视频元数据加载完成后设置 currentTime
      videoLeft.onloadedmetadata = () => {
        if (videoLeft.duration) {
          videoLeft.currentTime = currentPercent * videoLeft.duration;
        }
      };
      videoRight.onloadedmetadata = () => {
        if (videoRight.duration) {
          videoRight.currentTime = currentPercent * videoRight.duration;
        }
      };
    }
  
    loadCurrentGroup();
  
    // ✅ 拖动进度条控制逻辑
    let isDragging = false;
    let offsetX = 0;
  
    const containerRect = container.getBoundingClientRect();
    const globeRect = globe.getBoundingClientRect();
  
    const edgePadding = 6.5;
    const maxLeft = containerRect.width - globeRect.width - edgePadding;
    const minLeft = 0;
  
    function updateVideosToCurrentTime() {
      if (videoLeft.duration) {
        videoLeft.currentTime = currentPercent * videoLeft.duration;
      }
      if (videoRight.duration) {
        videoRight.currentTime = currentPercent * videoRight.duration;
      }
    }
  
    function startDrag(e) {
      isDragging = true;
      globe.style.cursor = 'grabbing';
  
      const clientX = e.type.includes('touch') ? e.touches[0].clientX : e.clientX;
      offsetX = clientX - globe.offsetLeft;
  
      globe.style.transition = 'none';
      wrapper.style.transition = 'none';
  
      videoLeft.pause();
      videoRight.pause();
    }
  
    function onDrag(e) {
      if (!isDragging) return;
  
      const clientX = e.type.includes('touch') ? e.touches[0].clientX : e.clientX;
      let newLeft = clientX - offsetX;
  
      newLeft = Math.max(minLeft, Math.min(newLeft, maxLeft));
      globe.style.left = newLeft + 'px';
  
      currentPercent = (newLeft - minLeft) / (maxLeft - minLeft);
      const wrapperLeft = -462 + currentPercent * 462;
      wrapper.style.left = `${wrapperLeft}px`;
  
      updateVideosToCurrentTime();
    }
  
    function stopDrag() {
      isDragging = false;
      globe.style.cursor = 'grab';
  
    //   videoLeft.play();
    //   videoRight.play();
    }
  
    globe.addEventListener('mousedown', startDrag);
    globe.addEventListener('touchstart', startDrag, { passive: true });
  
    document.addEventListener('mousemove', onDrag);
    document.addEventListener('touchmove', onDrag, { passive: false });
  
    document.addEventListener('mouseup', stopDrag);
    document.addEventListener('touchend', stopDrag);
  
    // ✅ 切换视频组
    function switchGroup(direction) {
      currentGroupIndex += direction;
      if (currentGroupIndex < 0) {
        currentGroupIndex = videoGroups.length - 1;
      } else if (currentGroupIndex >= videoGroups.length) {
        currentGroupIndex = 0;
      }
  
      loadCurrentGroup();
    }
  
    document.getElementById('prevGroup').addEventListener('click', () => switchGroup(-1));
    document.getElementById('nextGroup').addEventListener('click', () => switchGroup(1));
  });
  