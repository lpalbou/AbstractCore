# AbstractCore Navbar Component

This document explains how to use and maintain the shared navbar component across all AbstractCore documentation pages.

## Overview

The navbar component (`assets/js/navbar-component.js`) provides a consistent, animated navigation bar with the AbstractCore logo and configurable menu items. It ensures all pages have the same look and feel while allowing customization per page.

## Features

- **Animated Logo**: Consistent AbstractCore logo with pulse and float animations
- **Configurable Menu Items**: Up to 5 menu items can be configured per page
- **Responsive Design**: Mobile-friendly with hamburger menu
- **Auto-initialization**: Automatically detects and initializes on page load
- **Path-aware Links**: Automatically adjusts links based on page location (root vs docs)

## Usage

### 1. Include the Component

Add the navbar component script to your HTML head:

```html
<!-- Navbar Component -->
<script src="assets/js/navbar-component.js"></script>
<!-- OR for docs pages -->
<script src="../assets/js/navbar-component.js"></script>
```

### 2. Add the Placeholder

Replace your existing navbar HTML with a simple placeholder:

```html
<!-- Navigation -->
<div class="navbar-placeholder"></div>
```

### 3. Configure Menu Items (Optional)

For custom menu items, add a configuration script:

```html
<script>
document.addEventListener('DOMContentLoaded', function() {
    createNavbar({
        basePath: '../',  // For docs pages, use '' for root
        menuItems: [
            { text: 'Features', href: 'index.html#features' },
            { text: 'Quick Start', href: 'index.html#quickstart' },
            { text: 'Documentation', href: 'index.html#docs' },
            { text: 'Examples', href: 'index.html#examples' },
            { 
                text: 'GitHub', 
                href: 'https://github.com/lpalbou/AbstractCore',
                target: '_blank',
                icon: 'github'
            }
        ]
    });
});
</script>
```

## Configuration Options

### `basePath` (string)
- **Root pages**: `''` (empty string)
- **Docs pages**: `'../'`
- **Subdirectory pages**: Adjust as needed

### `menuItems` (array)
Each menu item can have:
- `text`: Display text for the menu item
- `href`: Link URL (can be relative, absolute, or hash)
- `target`: Optional target attribute (e.g., `'_blank'`)
- `icon`: Optional icon type (currently supports `'github'`)

## Default Configuration

If no configuration is provided, the component uses these defaults:

```javascript
{
    basePath: '',
    menuItems: [
        { text: 'Features', href: '#features' },
        { text: 'Quick Start', href: '#quickstart' },
        { text: 'Documentation', href: '#docs' },
        { text: 'Examples', href: '#examples' },
        { 
            text: 'PyPI', 
            href: 'https://pypi.org/project/abstractcore/',
            target: '_blank',
            icon: 'pypi'
        },
        { 
            text: 'GitHub', 
            href: 'https://github.com/lpalbou/AbstractCore',
            target: '_blank',
            icon: 'github'
        }
    ]
}
```

## Examples

### Root Page (index.html)
```html
<!DOCTYPE html>
<html>
<head>
    <script src="assets/js/navbar-component.js"></script>
</head>
<body>
    <div class="navbar-placeholder"></div>
    <!-- Component auto-initializes with default config -->
</body>
</html>
```

### Docs Page with Custom Menu
```html
<!DOCTYPE html>
<html>
<head>
    <script src="../assets/js/navbar-component.js"></script>
</head>
<body>
    <div class="navbar-placeholder"></div>
    
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        createNavbar({
            basePath: '../',
            menuItems: [
                { text: 'Home', href: 'index.html' },
                { text: 'API Docs', href: 'api-reference.html' },
                { text: 'GitHub', href: 'https://github.com/lpalbou/AbstractCore', target: '_blank', icon: 'github' }
            ]
        });
    });
    </script>
</body>
</html>
```

## Maintenance

### Updating the Logo
The logo animation and styling is defined in the `getNavbarCSS()` method. To update:

1. Edit `assets/js/navbar-component.js`
2. Modify the CSS in `getNavbarCSS()`
3. All pages will automatically use the updated logo

### Adding New Icons
To add support for new icons:

1. Create a new method like `getPyPIIcon()` or `getGitHubIcon()`
2. Update the menu item generation logic in `getMenuItemsHTML()` to handle the new icon type
3. Test across all pages

**Supported Icons:**
- `github`: GitHub logo
- `pypi`: PyPI logo

### Changing Default Menu Items
To change the default menu items:

1. Edit the `config` object in the `AbstractCoreNavbar` constructor
2. Update the `menuItems` array with new defaults
3. All pages without custom configuration will use the new defaults

## CSS Classes

The component generates these CSS classes:
- `.navbar`: Main navbar container
- `.nav-container`: Inner container with max-width
- `.nav-brand`: Logo container
- `.logo-container`: Animated logo wrapper
- `.logo-circle`: Main logo circle
- `.logo-lines`: Logo lines container
- `.logo-line`: Individual logo lines
- `.brand-text`: "AbstractCore" text
- `.nav-menu`: Menu items container
- `.nav-link`: Individual menu links
- `.nav-toggle`: Mobile hamburger menu
- `.github-icon`: GitHub icon styling

## Browser Support

The component works in all modern browsers that support:
- ES6 classes
- CSS animations
- Flexbox
- CSS Grid (for responsive layout)

## Troubleshooting

### Logo Not Appearing
1. Check that `navbar-component.js` is loaded
2. Verify the placeholder `<div class="navbar-placeholder"></div>` exists
3. Check browser console for JavaScript errors

### Menu Items Not Working
1. Verify the `createNavbar()` function is called after DOM load
2. Check that `basePath` is correct for the page location
3. Ensure menu item `href` values are valid

### Mobile Menu Not Working
1. Check that the viewport meta tag is present: `<meta name="viewport" content="width=device-width, initial-scale=1.0">`
2. Verify CSS is loading properly
3. Check for JavaScript errors in mobile browser console
