// MCR ML-VAR Documentation JavaScript

// DOM Elements
const themeToggle = document.getElementById('themeToggle');
const body = document.body;
const navLinks = document.querySelectorAll('.nav-link');
const copyButtons = document.querySelectorAll('.copy-btn');

// Theme Management
class ThemeManager {
  constructor() {
    this.currentTheme = this.getStoredTheme() || this.getSystemTheme();
    this.init();
  }

  init() {
    this.setTheme(this.currentTheme);
    this.bindEvents();
  }

  getSystemTheme() {
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  getStoredTheme() {
    try {
      return localStorage.getItem('theme');
    } catch (e) {
      console.warn('localStorage not available, using system theme');
      return null;
    }
  }

  setTheme(theme) {
    this.currentTheme = theme;
    document.documentElement.setAttribute('data-theme', theme);
    
    // Store theme preference
    try {
      localStorage.setItem('theme', theme);
    } catch (e) {
      console.warn('Could not save theme preference');
    }
  }

  toggleTheme() {
    const newTheme = this.currentTheme === 'dark' ? 'light' : 'dark';
    this.setTheme(newTheme);
  }

  bindEvents() {
    if (themeToggle) {
      themeToggle.addEventListener('click', () => {
        this.toggleTheme();
      });
    }

    // Listen for system theme changes
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
      if (!this.getStoredTheme()) {
        this.setTheme(e.matches ? 'dark' : 'light');
      }
    });
  }
}

// Copy to Clipboard Functionality
class ClipboardManager {
  constructor() {
    this.init();
  }

  init() {
    copyButtons.forEach(button => {
      button.addEventListener('click', (e) => {
        this.copyToClipboard(e.currentTarget);
      });
    });
  }

  async copyToClipboard(button) {
    const textToCopy = button.getAttribute('data-copy');
    
    if (!textToCopy) {
      console.error('No text to copy found');
      return;
    }

    try {
      // Try modern clipboard API first
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(textToCopy);
      } else {
        // Fallback for older browsers
        this.fallbackCopyTextToClipboard(textToCopy);
      }
      
      this.showCopySuccess(button);
    } catch (err) {
      console.error('Failed to copy text: ', err);
      this.showCopyError(button);
    }
  }

  fallbackCopyTextToClipboard(text) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.left = '-999999px';
    textArea.style.top = '-999999px';
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    
    try {
      document.execCommand('copy');
    } catch (err) {
      throw new Error('Fallback copy failed');
    } finally {
      document.body.removeChild(textArea);
    }
  }

  showCopySuccess(button) {
    const originalContent = button.innerHTML;
    button.classList.add('copied');
    button.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polyline points="20,6 9,17 4,12"/>
      </svg>
      Copied!
    `;
    
    setTimeout(() => {
      button.classList.remove('copied');
      button.innerHTML = originalContent;
    }, 2000);
  }

  showCopyError(button) {
    const originalContent = button.innerHTML;
    button.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10"/>
        <line x1="15" y1="9" x2="9" y2="15"/>
        <line x1="9" y1="9" x2="15" y2="15"/>
      </svg>
      Error
    `;
    
    setTimeout(() => {
      button.innerHTML = originalContent;
    }, 2000);
  }
}

// Navigation Manager
class NavigationManager {
  constructor() {
    this.sections = [];
    this.init();
  }

  init() {
    this.bindSmoothScrolling();
    this.bindActiveSection();
    this.handleInitialHash();
  }

  bindSmoothScrolling() {
    navLinks.forEach(link => {
      link.addEventListener('click', (e) => {
        const href = link.getAttribute('href');
        
        if (href.startsWith('#')) {
          e.preventDefault();
          const targetId = href.substring(1);
          const targetElement = document.getElementById(targetId);
          
          if (targetElement) {
            this.scrollToElement(targetElement);
            this.updateURL(href);
          }
        }
      });
    });
  }

  scrollToElement(element) {
    const headerHeight = document.querySelector('.header').offsetHeight;
    const targetPosition = element.offsetTop - headerHeight - 20;
    
    window.scrollTo({
      top: targetPosition,
      behavior: 'smooth'
    });
  }

  updateURL(hash) {
    if (history.pushState) {
      history.pushState(null, null, hash);
    } else {
      location.hash = hash;
    }
  }

  handleInitialHash() {
    if (window.location.hash) {
      const targetId = window.location.hash.substring(1);
      const targetElement = document.getElementById(targetId);
      
      if (targetElement) {
        // Delay scroll to ensure page is fully loaded
        setTimeout(() => {
          this.scrollToElement(targetElement);
        }, 100);
      }
    }
  }

  bindActiveSection() {
    // Get all sections with IDs
    const sections = document.querySelectorAll('section[id]');
    
    if (sections.length === 0) return;

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        const navLink = document.querySelector(`a[href="#${entry.target.id}"]`);
        
        if (entry.isIntersecting) {
          // Remove active class from all nav links
          navLinks.forEach(link => link.classList.remove('active'));
          
          // Add active class to current section's nav link
          if (navLink) {
            navLink.classList.add('active');
          }
        }
      });
    }, {
      rootMargin: '-100px 0px -80% 0px'
    });

    sections.forEach(section => {
      observer.observe(section);
    });
  }
}

// Performance and Loading Manager
class PerformanceManager {
  constructor() {
    this.init();
  }

  init() {
    this.lazyLoadImages();
    this.addLoadingStates();
    this.optimizeAnimations();
  }

  lazyLoadImages() {
    const images = document.querySelectorAll('img[data-src]');
    
    if ('IntersectionObserver' in window) {
      const imageObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            const img = entry.target;
            img.src = img.dataset.src;
            img.removeAttribute('data-src');
            imageObserver.unobserve(img);
          }
        });
      });

      images.forEach(img => imageObserver.observe(img));
    } else {
      // Fallback for older browsers
      images.forEach(img => {
        img.src = img.dataset.src;
        img.removeAttribute('data-src');
      });
    }
  }

  addLoadingStates() {
    // Add loading states to buttons that might trigger async actions
    const asyncButtons = document.querySelectorAll('[data-async]');
    
    asyncButtons.forEach(button => {
      button.addEventListener('click', () => {
        button.classList.add('loading');
        
        // Remove loading state after a reasonable time
        setTimeout(() => {
          button.classList.remove('loading');
        }, 2000);
      });
    });
  }

  optimizeAnimations() {
    // Respect user's preference for reduced motion
    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      document.documentElement.style.setProperty('--duration-fast', '0ms');
      document.documentElement.style.setProperty('--duration-normal', '0ms');
    }
  }
}

// Accessibility Manager
class AccessibilityManager {
  constructor() {
    this.init();
  }

  init() {
    this.enhanceKeyboardNavigation();
    this.addAriaLabels();
    this.manageFocusStates();
  }

  enhanceKeyboardNavigation() {
    // Handle escape key for closing modals/menus
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        // Close any open modals or menus
        const openElements = document.querySelectorAll('[data-open="true"]');
        openElements.forEach(element => {
          element.setAttribute('data-open', 'false');
        });
      }
    });

    // Enhance tab navigation
    const focusableElements = document.querySelectorAll(
      'a[href], button, textarea, input, select, [tabindex]:not([tabindex="-1"])'
    );

    focusableElements.forEach(element => {
      element.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && element.tagName === 'A') {
          element.click();
        }
      });
    });
  }

  addAriaLabels() {
    // Add missing aria-labels where needed
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle && !themeToggle.getAttribute('aria-label')) {
      themeToggle.setAttribute('aria-label', 'Toggle dark mode');
    }

    // Add aria-labels to copy buttons
    copyButtons.forEach((button, index) => {
      if (!button.getAttribute('aria-label')) {
        button.setAttribute('aria-label', `Copy code block ${index + 1}`);
      }
    });
  }

  manageFocusStates() {
    // Add visible focus indicators for keyboard users
    let isUsingKeyboard = false;

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Tab') {
        isUsingKeyboard = true;
        document.body.classList.add('using-keyboard');
      }
    });

    document.addEventListener('mousedown', () => {
      isUsingKeyboard = false;
      document.body.classList.remove('using-keyboard');
    });
  }
}

// Utility Functions
const utils = {
  // Debounce function for performance optimization
  debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  },

  // Throttle function for scroll events
  throttle(func, limit) {
    let inThrottle;
    return function(...args) {
      if (!inThrottle) {
        func.apply(this, args);
        inThrottle = true;
        setTimeout(() => inThrottle = false, limit);
      }
    };
  },

  // Check if element is in viewport
  isInViewport(element) {
    const rect = element.getBoundingClientRect();
    return (
      rect.top >= 0 &&
      rect.left >= 0 &&
      rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
      rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
  }
};

// Initialize Application
class App {
  constructor() {
    this.init();
  }

  init() {
    // Wait for DOM to be fully loaded
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => {
        this.initializeManagers();
      });
    } else {
      this.initializeManagers();
    }
  }

  initializeManagers() {
    try {
      // Initialize all managers
      this.themeManager = new ThemeManager();
      this.clipboardManager = new ClipboardManager();
      this.navigationManager = new NavigationManager();
      this.performanceManager = new PerformanceManager();
      this.accessibilityManager = new AccessibilityManager();

      // Add global event listeners
      this.bindGlobalEvents();
      
      console.log('MCR ML-VAR Documentation initialized successfully');
    } catch (error) {
      console.error('Error initializing application:', error);
    }
  }

  bindGlobalEvents() {
    // Handle window resize
    window.addEventListener('resize', utils.throttle(() => {
      // Handle responsive behavior
      this.handleResize();
    }, 250));

    // Handle scroll events
    window.addEventListener('scroll', utils.throttle(() => {
      this.handleScroll();
    }, 100));

    // Handle visibility change (for performance optimization)
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        // Page is hidden, pause any running animations or timers
        this.pauseAnimations();
      } else {
        // Page is visible, resume animations
        this.resumeAnimations();
      }
    });
  }

  handleResize() {
    // Handle responsive navigation
    const nav = document.querySelector('.nav');
    const headerContent = document.querySelector('.header-content');
    
    if (window.innerWidth <= 768) {
      nav.classList.add('mobile-nav');
    } else {
      nav.classList.remove('mobile-nav');
    }
  }

  handleScroll() {
    const header = document.querySelector('.header');
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;

    // Add/remove scrolled class for header styling
    if (scrollTop > 100) {
      header.classList.add('scrolled');
    } else {
      header.classList.remove('scrolled');
    }
  }

  pauseAnimations() {
    document.body.classList.add('animations-paused');
  }

  resumeAnimations() {
    document.body.classList.remove('animations-paused');
  }
}

// Initialize the application
const app = new App();

// Export for potential external use
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { App, ThemeManager, ClipboardManager, NavigationManager };
}