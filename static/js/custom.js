document.addEventListener('DOMContentLoaded', function () {
    const heroSection = document.getElementById('hero-section');
    let isScrolling = false; // 防止短时间内重复触发
    let startY = 0; // 记录触摸起始位置
  
    // 判断 hero section 是否在视口中（部分可见即可）
    function isInHeroSection() {
      const rect = heroSection.getBoundingClientRect();
      return rect.top < window.innerHeight && rect.bottom > 0;
    }
  
    // 滚动到下一屏
    function scrollToNextSection() {
      isScrolling = true;
      window.scrollTo({
        top: window.innerHeight,
        behavior: 'smooth'
      });
  
      // 设置防抖时间，防止重复触发
      setTimeout(() => {
        isScrolling = false;
      }, 1000); // 1秒内不再触发
    }
  
    // 桌面端滚轮事件
    heroSection.addEventListener('wheel', function (e) {
      if (e.deltaY > 0 && !isScrolling && isInHeroSection()) {
        scrollToNextSection();
      }
    }, { passive: true });
  
    // 移动端触摸开始
    heroSection.addEventListener('touchstart', function (e) {
      if (e.touches.length === 1) {
        startY = e.touches[0].pageY;
      }
    }, { passive: true });
  
    // 移动端触摸滑动
    heroSection.addEventListener('touchmove', function (e) {
      if (e.touches.length === 1 && !isScrolling && isInHeroSection()) {
        const currentY = e.touches[0].pageY;
        const diff = startY - currentY;
  
        if (diff > 50) { // 向上滑动超过 50px（页面向下滚）
          scrollToNextSection();
        }
      }
    }, { passive: true });
  });
  