import type {
  LicenseConfig,
  NavBarConfig,
  ProfileConfig,
  SiteConfig,
} from './types/config'
import { LinkPreset } from './types/config'

export const siteConfig: SiteConfig = {
  title: 'Cohen Gray Archbold',
  subtitle: 'Personal & Profolio',
  lang: 'en',         // 'en', 'zh_CN', 'zh_TW', 'ja'
  themeColor: {
    hue: 170,         // Default hue for the theme color, from 0 to 360. e.g. red: 0, teal: 200, cyan: 250, pink: 345
    fixed: true,     // Hide the theme color picker for visitors
  },
  banner: {
    enable: true,
    src: 'assets/images/forest_banner.jpg',   // Relative to the /src directory. Relative to the /public directory if it starts with '/'
    position: 'top', // Equivalent to object-position, defaults center
  },
  favicon: [    // Leave this array empty to use the default favicon
    // {
    //   src: '/favicon/icon.png',    // Path of the favicon, relative to the /public directory
    //   theme: 'light',              // (Optional) Either 'light' or 'dark', set only if you have different favicons for light and dark mode
    //   sizes: '32x32',              // (Optional) Size of the favicon, set only if you have favicons of different sizes
    // }
  ]
}

export const navBarConfig: NavBarConfig = {
  links: [
    LinkPreset.Home,
    LinkPreset.Archive,
    LinkPreset.About,
    LinkPreset.CV,
    {
      name: 'GitHub',
      url: 'https://github.com/cgarchbold',     // Internal links should not include the base path, as it is automatically added
      external: true,                               // Show an external link icon and will open in a new tab
    },
  ],
}

export const profileConfig: ProfileConfig = {
  avatar: 'assets/images/Headshot_2023.JPG',  // Relative to the /src directory. Relative to the /public directory if it starts with '/'
  name: 'Cohen Gray Archbold',
  bio: 'PhD Student at the University of Kentucky. Researching Privacy and Fairness in Machine Learning for vision.',
  links: [
    {
      name: 'Linkedin',
      icon: 'fa6-brands:linkedin',
      url: 'https://www.linkedin.com/in/cgarchbold/',
    },
    {
      name: 'GitHub',
      icon: 'fa6-brands:github',
      url: 'https://github.com/cgarchbold',
    },
    {
      name: 'Discord',
      icon: 'fa6-brands:discord',
      url: 'https://discordapp.com/users/coheen',
    },
  ],
}

export const licenseConfig: LicenseConfig = {
  enable: true,
  name: 'CC BY-NC-SA 4.0',
  url: 'https://creativecommons.org/licenses/by-nc-sa/4.0/',
}
