
import { Fragment, useState } from 'react'
import { Dialog, Transition } from '@headlessui/react'
import { Link, useLocation, Outlet } from 'react-router-dom'
import { 
  Activity, 
  Menu, 
  X, 
  Database,
  BarChart2,
  Settings,
  Cpu
} from 'lucide-react'
import clsx from 'clsx'

const navigation = [
  { name: 'Analysis Workbench', href: '/', icon: Activity },
  { name: 'Model Metrics', href: '#', icon: BarChart2, current: false },
  { name: 'Data Export', href: '#', icon: Database, current: false },
  { name: 'System Config', href: '#', icon: Settings, current: false },
]

export default function DashboardLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const location = useLocation()

  return (
    <>
      <div>
        <Transition.Root show={sidebarOpen} as={Fragment}>
          <Dialog as="div" className="relative z-50 lg:hidden" onClose={setSidebarOpen}>
            <Transition.Child
              as={Fragment}
              enter="transition-opacity ease-linear duration-300"
              enterFrom="opacity-0"
              enterTo="opacity-100"
              leave="transition-opacity ease-linear duration-300"
              leaveFrom="opacity-100"
              leaveTo="opacity-0"
            >
              <div className="fixed inset-0 bg-gray-900/80" />
            </Transition.Child>

            <div className="fixed inset-0 flex">
              <Transition.Child
                as={Fragment}
                enter="transition ease-in-out duration-300 transform"
                enterFrom="-translate-x-full"
                enterTo="translate-x-0"
                leave="transition ease-in-out duration-300 transform"
                leaveFrom="translate-x-0"
                leaveTo="-translate-x-full"
              >
                <Dialog.Panel className="relative mr-16 flex w-full max-w-xs flex-1">
                  <div className="flex grow flex-col gap-y-5 overflow-y-auto bg-white px-6 pb-4 ring-1 ring-gray-200">
                    <div className="flex h-16 shrink-0 items-center">
                      <div className="flex items-center gap-2">
                        <Cpu className="h-8 w-8 text-emerald-600" />
                        <span className="text-lg font-bold text-gray-900">MedGemma Research</span>
                      </div>
                    </div>
                    <nav className="flex flex-1 flex-col">
                      <ul role="list" className="flex flex-1 flex-col gap-y-7">
                        <li>
                          <ul role="list" className="-mx-2 space-y-1">
                            {navigation.map((item) => (
                              <li key={item.name}>
                                <Link
                                  to={item.href}
                                  className={clsx(
                                    location.pathname === item.href
                                      ? 'bg-emerald-50 text-emerald-600'
                                      : 'text-gray-700 hover:text-emerald-600 hover:bg-gray-50',
                                    'group flex gap-x-3 rounded-md p-2 text-sm leading-6 font-semibold'
                                  )}
                                >
                                  <item.icon
                                    className={clsx(
                                      location.pathname === item.href ? 'text-emerald-600' : 'text-gray-400 group-hover:text-emerald-600',
                                      'h-6 w-6 shrink-0'
                                    )}
                                    aria-hidden="true"
                                  />
                                  {item.name}
                                </Link>
                              </li>
                            ))}
                          </ul>
                        </li>
                      </ul>
                    </nav>
                  </div>
                </Dialog.Panel>
              </Transition.Child>
            </div>
          </Dialog>
        </Transition.Root>

        {/* Static sidebar for desktop */}
        <div className="hidden lg:fixed lg:inset-y-0 lg:z-50 lg:flex lg:w-72 lg:flex-col">
          <div className="flex grow flex-col gap-y-5 overflow-y-auto bg-white px-6 pb-4 border-r border-gray-200">
            <div className="flex h-16 shrink-0 items-center">
              <div className="flex items-center gap-3">
                <div className="bg-emerald-50 p-2 rounded-lg">
                    <Cpu className="h-6 w-6 text-emerald-600" />
                </div>
                <div>
                    <span className="block text-lg font-bold text-gray-900 tracking-tight leading-none">MedGemma</span>
                    <span className="text-xs text-gray-500 font-medium">Research Environment</span>
                </div>
              </div>
            </div>
            <nav className="flex flex-1 flex-col">
              <ul role="list" className="flex flex-1 flex-col gap-y-7">
                <li>
                    <div className="text-xs font-semibold leading-6 text-gray-400 uppercase tracking-wider mb-4">Tools</div>
                  <ul role="list" className="-mx-2 space-y-1">
                    {navigation.map((item) => (
                      <li key={item.name}>
                        <Link
                          to={item.href}
                          className={clsx(
                            location.pathname === item.href
                              ? 'bg-emerald-50 text-emerald-600'
                              : 'text-gray-600 hover:text-emerald-600 hover:bg-gray-50',
                            'group flex gap-x-3 rounded-xl p-3 text-sm leading-6 font-medium transition-all duration-200'
                          )}
                        >
                          <item.icon
                            className={clsx(
                              location.pathname === item.href ? 'text-emerald-600' : 'text-gray-400 group-hover:text-emerald-600',
                              'h-5 w-5 shrink-0'
                            )}
                            aria-hidden="true"
                          />
                          {item.name}
                        </Link>
                      </li>
                    ))}
                  </ul>
                </li>
                
                <li className="mt-auto">
                    <div className="rounded-xl bg-gray-50 p-4 border border-gray-100">
                        <div className="flex items-center gap-3">
                            <div className="h-10 w-10 rounded-full bg-emerald-100 flex items-center justify-center text-emerald-700 font-bold">
                                AI
                            </div>
                            <div>
                                <div className="text-sm font-semibold text-gray-900">Model v2.1</div>
                                <div className="text-xs text-gray-500">ResNet3D Backbone</div>
                            </div>
                        </div>
                    </div>
                </li>
              </ul>
            </nav>
          </div>
        </div>

        <div className="lg:pl-72 bg-gray-50 min-h-screen">
          <div className="sticky top-0 z-40 flex h-16 shrink-0 items-center gap-x-4 border-b border-gray-200 bg-white px-4 shadow-sm sm:gap-x-6 sm:px-6 lg:px-8">
            <button
              type="button"
              className="-m-2.5 p-2.5 text-gray-700 lg:hidden"
              onClick={() => setSidebarOpen(true)}
            >
              <span className="sr-only">Open sidebar</span>
              <Menu className="h-6 w-6" aria-hidden="true" />
            </button>

            {/* Separator */}
            <div className="h-6 w-px bg-gray-200 lg:hidden" aria-hidden="true" />

            <div className="flex flex-1 gap-x-4 self-stretch lg:gap-x-6">
               <div className="flex items-center gap-2 text-sm text-gray-500">
                    <Activity className="h-4 w-4" />
                    <span>/</span>
                    <span className="text-gray-900 font-medium">
                        Workbench
                    </span>
               </div>
               <div className="ml-auto flex items-center gap-2">
                    <span className="inline-flex items-center rounded-md bg-emerald-50 px-2 py-1 text-xs font-medium text-emerald-700 ring-1 ring-inset ring-emerald-600/20">
                        GPU Active
                    </span>
               </div>
            </div>
          </div>

          <main className="py-10">
            <div className="px-4 sm:px-6 lg:px-8">
              <Outlet />
            </div>
          </main>
        </div>
      </div>
    </>
  )
}
