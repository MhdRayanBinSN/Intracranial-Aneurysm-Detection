import { useState, useEffect } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Bars3Icon, XMarkIcon } from '@heroicons/react/24/outline'
import clsx from 'clsx'

const navItems = [
  { name: 'Home', path: '/' },
  { name: 'Analysis', path: '/analysis' },
  { name: 'MedGemma', path: '/medgemma' },
  { name: 'Compare', path: '/compare' },
  { name: 'Dataset', path: '/dataset' },
]

export default function Navbar() {
  const [isScrolled, setIsScrolled] = useState(false)
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)
  const location = useLocation()

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10)
    }
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  return (
    <nav
      className={clsx(
        'fixed top-0 left-0 right-0 z-50 transition-all duration-300',
        isScrolled
          ? 'bg-surface-950/90 backdrop-blur-xl border-b border-surface-800'
          : 'bg-transparent'
      )}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center space-x-3">
            <motion.div
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="w-10 h-10 bg-gradient-to-br from-primary-500 to-accent-500 rounded-xl flex items-center justify-center"
            >
              <svg
                className="w-6 h-6 text-white"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                />
              </svg>
            </motion.div>
            <span className="text-xl font-bold text-white">
              Aneurysm<span className="text-primary-400">Detector</span>
            </span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-1">
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={clsx(
                  'px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200',
                  location.pathname === item.path
                    ? 'bg-primary-500/20 text-primary-400'
                    : 'text-surface-300 hover:text-white hover:bg-surface-800'
                )}
              >
                {item.name}
              </Link>
            ))}
          </div>

          {/* CTA Button */}
          <div className="hidden md:block">
            <Link to="/analysis">
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="btn-primary"
              >
                Start Analysis
              </motion.button>
            </Link>
          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            className="md:hidden p-2 rounded-lg text-surface-400 hover:text-white hover:bg-surface-800"
          >
            {isMobileMenuOpen ? (
              <XMarkIcon className="w-6 h-6" />
            ) : (
              <Bars3Icon className="w-6 h-6" />
            )}
          </button>
        </div>
      </div>

      {/* Mobile Menu */}
      <motion.div
        initial={false}
        animate={{
          height: isMobileMenuOpen ? 'auto' : 0,
          opacity: isMobileMenuOpen ? 1 : 0,
        }}
        className="md:hidden overflow-hidden bg-surface-900/95 backdrop-blur-xl border-b border-surface-800"
      >
        <div className="px-4 py-4 space-y-2">
          {navItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              onClick={() => setIsMobileMenuOpen(false)}
              className={clsx(
                'block px-4 py-3 rounded-lg text-sm font-medium',
                location.pathname === item.path
                  ? 'bg-primary-500/20 text-primary-400'
                  : 'text-surface-300 hover:text-white hover:bg-surface-800'
              )}
            >
              {item.name}
            </Link>
          ))}
          <Link
            to="/analysis"
            onClick={() => setIsMobileMenuOpen(false)}
            className="block"
          >
            <button className="w-full btn-primary mt-2">Start Analysis</button>
          </Link>
        </div>
      </motion.div>
    </nav>
  )
}
