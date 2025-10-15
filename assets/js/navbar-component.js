/**
 * Reusable Navbar Component for AbstractCore
 * 
 * Creates a consistent navbar with animated logo and configurable menu items
 */

class AbstractCoreNavbar {
    constructor(config = {}) {
        this.config = {
            // Base path for links (e.g., '../' for docs pages, '' for root)
            basePath: config.basePath || '',
            // Menu items configuration
            menuItems: config.menuItems || [
                { text: 'Features', href: '#features' },
                { text: 'Quick Start', href: '#quickstart' },
                { text: 'Documentation', href: '#docs' },
                { text: 'Examples', href: '#examples' },
                { 
                    text: 'GitHub', 
                    href: 'https://github.com/lpalbou/AbstractCore',
                    target: '_blank',
                    icon: 'github'
                }
            ]
        };
    }

    /**
     * Generate the animated AbstractCore logo HTML
     */
    getLogoHTML() {
        const logoHref = this.config.basePath ? `${this.config.basePath}index.html` : '#';
        
        return `
            <a href="${logoHref}" class="logo-link">
                <div class="logo-container">
                    <div class="logo-circle"></div>
                    <div class="logo-lines">
                        <div class="logo-line"></div>
                        <div class="logo-line"></div>
                        <div class="logo-line"></div>
                    </div>
                </div>
                <span class="brand-text">AbstractCore</span>
            </a>
        `;
    }

    /**
     * Generate GitHub icon SVG
     */
    getGitHubIcon() {
        return `
            <svg class="github-icon" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.30.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
            </svg>
        `;
    }

    /**
     * Generate menu items HTML
     */
    getMenuItemsHTML() {
        return this.config.menuItems.map(item => {
            const href = item.href.startsWith('#') || item.href.startsWith('http') 
                ? item.href 
                : `${this.config.basePath}${item.href}`;
            
            const target = item.target ? `target="${item.target}"` : '';
            const icon = item.icon === 'github' ? this.getGitHubIcon() : '';
            
            return `
                <a href="${href}" class="nav-link" ${target}>
                    ${icon}
                    ${item.text}
                </a>
            `;
        }).join('');
    }

    /**
     * Generate complete navbar HTML
     */
    getNavbarHTML() {
        return `
            <nav class="navbar">
                <div class="nav-container">
                    <div class="nav-brand">
                        ${this.getLogoHTML()}
                    </div>
                    
                    <div class="nav-menu" id="nav-menu">
                        ${this.getMenuItemsHTML()}
                    </div>
                    
                    <div class="nav-toggle" id="nav-toggle">
                        <span class="bar"></span>
                        <span class="bar"></span>
                        <span class="bar"></span>
                    </div>
                </div>
            </nav>
        `;
    }

    /**
     * Get the CSS styles for the navbar
     */
    getNavbarCSS() {
        return `
            .navbar {
                background: rgba(15, 23, 42, 0.95);
                backdrop-filter: blur(10px);
                border-bottom: 1px solid rgba(148, 163, 184, 0.1);
                position: fixed;
                top: 0;
                width: 100%;
                z-index: 1000;
                transition: all 0.3s ease;
            }

            .nav-container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 2rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
                height: 70px;
            }

            .nav-brand {
                display: flex;
                align-items: center;
            }

            .logo-link {
                display: flex;
                align-items: center;
                text-decoration: none;
                color: inherit;
            }

            .logo-container {
                position: relative;
                width: 40px;
                height: 40px;
                margin-right: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
            }

            .logo-circle {
                width: 32px;
                height: 32px;
                border-radius: 50%;
                background: linear-gradient(135deg, #06b6d4, #3b82f6);
                position: relative;
                animation: pulse 2s ease-in-out infinite;
            }

            .logo-lines {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: 16px;
                height: 16px;
            }

            .logo-line {
                width: 100%;
                height: 2px;
                background: white;
                margin: 2px 0;
                border-radius: 1px;
                animation: float 3s ease-in-out infinite;
            }

            .logo-line:nth-child(2) {
                animation-delay: 0.5s;
            }

            .logo-line:nth-child(3) {
                animation-delay: 1s;
            }

            .brand-text {
                font-size: 1.5rem;
                font-weight: 700;
                background: linear-gradient(135deg, #06b6d4, #3b82f6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }

            .nav-menu {
                display: flex;
                align-items: center;
                gap: 2rem;
            }

            .nav-link {
                color: #e2e8f0;
                text-decoration: none;
                font-weight: 500;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.5rem 1rem;
                border-radius: 0.5rem;
            }

            .nav-link:hover {
                color: #06b6d4;
                background: rgba(6, 182, 212, 0.1);
            }

            .github-icon {
                width: 20px;
                height: 20px;
            }

            .nav-toggle {
                display: none;
                flex-direction: column;
                cursor: pointer;
            }

            .bar {
                width: 25px;
                height: 3px;
                background: #e2e8f0;
                margin: 3px 0;
                transition: 0.3s;
                border-radius: 2px;
            }

            @keyframes pulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.05); }
            }

            @keyframes float {
                0%, 100% { transform: translateY(0px); }
                50% { transform: translateY(-2px); }
            }

            @media (max-width: 768px) {
                .nav-menu {
                    position: fixed;
                    left: -100%;
                    top: 70px;
                    flex-direction: column;
                    background: rgba(15, 23, 42, 0.98);
                    width: 100%;
                    text-align: center;
                    transition: 0.3s;
                    box-shadow: 0 10px 27px rgba(0, 0, 0, 0.05);
                    backdrop-filter: blur(10px);
                    padding: 2rem 0;
                    gap: 1rem;
                }

                .nav-menu.active {
                    left: 0;
                }

                .nav-toggle {
                    display: flex;
                }

                .nav-toggle.active .bar:nth-child(2) {
                    opacity: 0;
                }

                .nav-toggle.active .bar:nth-child(1) {
                    transform: translateY(8px) rotate(45deg);
                }

                .nav-toggle.active .bar:nth-child(3) {
                    transform: translateY(-8px) rotate(-45deg);
                }
            }
        `;
    }

    /**
     * Initialize the navbar on a page
     */
    init() {
        // Inject CSS if not already present
        if (!document.querySelector('#navbar-styles')) {
            const style = document.createElement('style');
            style.id = 'navbar-styles';
            style.textContent = this.getNavbarCSS();
            document.head.appendChild(style);
        }

        // Find navbar placeholder and inject HTML
        const placeholder = document.querySelector('.navbar-placeholder');
        if (placeholder) {
            placeholder.outerHTML = this.getNavbarHTML();
        } else {
            // If no placeholder, inject at the beginning of body
            document.body.insertAdjacentHTML('afterbegin', this.getNavbarHTML());
        }

        // Initialize mobile menu toggle
        this.initMobileMenu();
    }

    /**
     * Initialize mobile menu functionality
     */
    initMobileMenu() {
        const navToggle = document.getElementById('nav-toggle');
        const navMenu = document.getElementById('nav-menu');

        if (navToggle && navMenu) {
            navToggle.addEventListener('click', () => {
                navToggle.classList.toggle('active');
                navMenu.classList.toggle('active');
            });
        }
    }
}

/**
 * Convenience function to create and initialize navbar
 */
function createNavbar(config = {}) {
    const navbar = new AbstractCoreNavbar(config);
    navbar.init();
    return navbar;
}

// Auto-initialize if DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        // Only auto-init if there's a placeholder
        if (document.querySelector('.navbar-placeholder')) {
            createNavbar();
        }
    });
} else {
    // DOM already loaded
    if (document.querySelector('.navbar-placeholder')) {
        createNavbar();
    }
}

// Export for manual initialization
window.AbstractCoreNavbar = AbstractCoreNavbar;
window.createNavbar = createNavbar;
