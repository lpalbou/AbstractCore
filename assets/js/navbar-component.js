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
                { text: 'Features', href: '/docs/capabilities.html' },
                { text: 'Quick Start', href: '/docs/getting-started.html' },
                { text: 'Documentation', href: '/#docs' },
                { text: 'Examples', href: '/docs/examples.html' },
                {
                    text: 'GitHub',
                    href: 'https://github.com/lpalbou/AbstractCore',
                    target: '_blank',
                    icon: 'github'
                },
                {
                    text: 'PyPI',
                    href: 'https://pypi.org/project/abstractcore/',
                    target: '_blank',
                    icon: 'pypi'
                }
            ]
        };
    }

    /**
     * Generate the animated AbstractCore logo HTML
     */
    getLogoHTML() {
        const logoHref = 'https://lpalbou.github.io/AbstractCore/';
        
        return `
            <a href="${logoHref}" class="brand-link">
                <div class="logo-abstract">
                    <div class="logo-circle"></div>
                    <div class="orbit-container">
                        <div class="orbit-dot orbit-dot-1"></div>
                        <div class="orbit-dot orbit-dot-2"></div>
                    </div>
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
     * Generate PyPI icon SVG
     */
    getPyPIIcon() {
        return `
            <svg class="pypi-icon" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0C5.373 0 0 5.373 0 12s5.373 12 12 12 12-5.373 12-12S18.627 0 12 0zm-1.5 6h3c.825 0 1.5.675 1.5 1.5v3c0 .825-.675 1.5-1.5 1.5H12v1.5h3V15H9V7.5c0-.825.675-1.5 1.5-1.5zm0 1.5v3h3v-3h-3z"/>
            </svg>
        `;
    }

    /**
     * Generate menu items HTML
     */
    getMenuItemsHTML() {
        return this.config.menuItems.map(item => {
            // Don't prepend basePath if href is:
            // - An anchor (#something)
            // - An absolute URL (http/https)
            // - Already contains '../' or '/' (relative/absolute path)
            const href = item.href.startsWith('#') ||
                         item.href.startsWith('http') ||
                         item.href.startsWith('../') ||
                         item.href.startsWith('/')
                ? item.href
                : `${this.config.basePath}${item.href}`;

            const target = item.target ? `target="${item.target}"` : '';
            let icon = '';
            if (item.icon === 'github') {
                icon = this.getGitHubIcon();
            } else if (item.icon === 'pypi') {
                icon = this.getPyPIIcon();
            }

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
     * Get the CSS styles for the navbar (uses existing main.css styles)
     */
    getNavbarCSS() {
        return `
            /* Additional PyPI icon styles */
            .pypi-icon {
                width: 1.25rem;
                height: 1.25rem;
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
